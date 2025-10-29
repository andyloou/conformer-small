import argparse
import os
import glob
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from vai_q_onnx import quantize_static, VitisQuantFormat, PowerOfTwoMethod, CalibraterBase
import math
import librosa
import traceback
import vai_q_onnx
from datasets import load_dataset 
import soundfile as sf 
import gc


class FilterbankFeatures(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000, n_window_size=400, n_window_stride=160, window="hann",
        normalize="per_feature", n_fft=512, preemph=0.97, nfilt=80,
        lowfreq=0, highfreq=None, log=True, log_zero_guard_type="add",
        log_zero_guard_value=2**-24, dither=1e-5, pad_to=0,
        frame_splicing=1, pad_value=0, mag_power=2.0, mel_norm="slaney",
    ):
        super().__init__()
        if highfreq is None: highfreq = sample_rate / 2
        self.preemph, self.n_fft, self.nfilt = preemph, n_fft or n_window_size, nfilt
        self.normalize, self.log, self.dither = normalize, log, dither
        self.frame_splicing = frame_splicing
        self.n_window_size, self.n_window_stride = n_window_size, n_window_stride
        self.pad_to, self.pad_value = pad_to, pad_value
        self.log_zero_guard_type, self.log_zero_guard_value = log_zero_guard_type, log_zero_guard_value
        self.mag_power, self.mel_norm = mag_power, mel_norm
        torch_windows = {'hann': torch.hann_window, 'ones': torch.ones}
        self.register_buffer("window", torch_windows[window](n_window_size, periodic=False))
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, htk=False, norm=mel_norm)
        self.register_buffer("fb", torch.from_numpy(mel_basis).float())

    @torch.no_grad()
    def forward(self, audio, length):
        if self.dither > 0: audio += self.dither * torch.randn_like(audio)
        if self.preemph is not None:
            audio = torch.cat((audio[:, 0].unsqueeze(1), audio[:, 1:] - self.preemph * audio[:, :-1]), dim=1)
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.n_window_stride,
                          win_length=self.n_window_size, window=self.window.to(audio.device),
                          center=True, pad_mode='reflect', return_complex=True)
        mag = torch.abs(stft)
        power = mag.pow(self.mag_power)
        mel = torch.matmul(self.fb.to(power.device), power)
        if self.log:
            mel = torch.log(torch.clamp(mel, min=self.log_zero_guard_value))
        if self.normalize == "per_feature":
            mean, std = torch.mean(mel, dim=-1, keepdim=True), torch.std(mel, dim=-1, keepdim=True)
            mel = (mel - mean) / (std + 1e-5)
        length = (length + self.n_window_stride // 2) // self.n_window_stride
        return mel, length


class AudioToMelSpectrogramPreprocessor(torch.nn.Module):
    # ... (Code giữ nguyên) ...
    def __init__( self, sample_rate=16000, window_size=0.025, window_stride=0.01,
        window="hann", normalize="per_feature", n_fft=512, features=80, **kwargs):
        super().__init__()
        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate, n_window_size=int(window_size * sample_rate),
            n_window_stride=int(window_stride * sample_rate), window=window,
            normalize=normalize, n_fft=n_fft, nfilt=features, **kwargs
        )
    @torch.no_grad()
    def forward(self, input_signal, length):
        return self.featurizer(input_signal, length)


class MelSpecDataReader(CalibraterBase):
    def __init__(
        self, 
        model_path, 
        dataset_path=None,  # Giữ để backward compatible
        max_samples=100, 
        batch_size=1, 
        input_name='mel_spectrogram', 
        length_name='mel_length',
        use_huggingface=True,  # NEW: Flag để chọn source
        dataset_name="linhtran92/viet_bud500",  # NEW
        split="test",  # NEW
        max_duration=20.0  # NEW: Giới hạn audio dài
    ):
        super().__init__(model_path)
        print("Initializing Preprocessor for Calibration...")
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=16000, window_size=0.025, window_stride=0.01,
            features=80, n_fft=512, dither=1.0e-05, preemph=0.97
        )
        self.preprocessor.eval()
        print("Preprocessor Initialized.")

        self.use_huggingface = use_huggingface
        self.max_duration = max_duration
        self.target_sr = 16000

        if use_huggingface:
            # Load từ HuggingFace
            print(f"Loading HuggingFace dataset: {dataset_name} (split={split})")
            self.dataset = load_dataset(dataset_name, split=split, streaming=False)
            
            # Filter audio dài (nếu cần)
            if max_duration:
                print(f"Filtering audios with duration <= {max_duration}s")
                def filter_fn(example):
                    duration = len(example['audio']['array']) / example['audio']['sampling_rate']
                    return duration <= max_duration
                self.dataset = self.dataset.filter(filter_fn)
            
            # Giới hạn số lượng samples
            total_samples = len(self.dataset)
            if max_samples > 0 and total_samples > max_samples:
                print(f"Limiting calibration to {max_samples} random samples (total={total_samples})")
                indices = np.random.choice(total_samples, max_samples, replace=False)
                self.dataset = self.dataset.select(indices)
            
            print(f"Prepared {len(self.dataset)} samples for calibration.")
            self.audio_files = None  # Không dùng file paths
        else:
            # Load từ local files (code cũ)
            print(f"Scanning LOCAL dataset at: {dataset_path}")
            self.audio_files = sorted(glob.glob(os.path.join(dataset_path, "**", "*.flac"), recursive=True))
            
            if not self.audio_files:
                print("No .flac files found, searching for .wav files...")
                self.audio_files = sorted(glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True))
            
            if not self.audio_files:
                raise FileNotFoundError(f"No .flac or .wav files found in {dataset_path}")

            if max_samples > 0 and len(self.audio_files) > max_samples:
                print(f"Limiting calibration to {max_samples} random samples.")
                indices = np.random.choice(len(self.audio_files), max_samples, replace=False)
                self.audio_files = [self.audio_files[i] for i in indices]
            
            print(f"Found {len(self.audio_files)} files for calibration.")
            self.dataset = None

        self.input_name = input_name
        self.length_name = length_name
        self.batch_size = 1
        self.enum_data = None

    def get_next(self):
        """Returns the next batch of data (batch size is 1)."""
        if self.enum_data is None:
            self.enum_data = iter(self._process_data())
        
        batch_data = next(self.enum_data, None)
        return batch_data

    def _process_data(self):
        """Generator function to process audio samples."""
        if self.use_huggingface:
            # Process HuggingFace dataset
            for i, example in enumerate(self.dataset):
                try:
                    # Extract audio từ HuggingFace format
                    audio_dict = example['audio']
                    waveform_np = audio_dict['array']  # NumPy array
                    sr = audio_dict['sampling_rate']
                    
                    # Convert to torch tensor [1, T]
                    waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
                    
                    # Resample nếu cần
                    if sr != self.target_sr:
                        waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
                    
                    # Convert to mono nếu cần
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Get length
                    length = torch.tensor([waveform.shape[1]], dtype=torch.long)
                    
                    # Run preprocessor
                    mel_tensor, mel_length_tensor = self.preprocessor(waveform, length)
                    
                    # Convert to numpy
                    mel_np = mel_tensor.cpu().numpy().astype(np.float32)
                    mel_length_np = mel_length_tensor.cpu().numpy().astype(np.int64)
                    
                    yield {self.input_name: mel_np, self.length_name: mel_length_np}
                
                except Exception as e:
                    print(f"\nError processing HF sample {i}: {e}. Skipping.")
                    continue
        else:
            # Process local files (code cũ)
            for i, audio_path in enumerate(self.audio_files):
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    if sr != self.target_sr:
                        waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
                    if waveform.dtype != torch.float32:
                        waveform = waveform.float()
                    
                    length = torch.tensor([waveform.shape[1]], dtype=torch.long)
                    
                    mel_tensor, mel_length_tensor = self.preprocessor(waveform, length)
                    
                    mel_np = mel_tensor.cpu().numpy().astype(np.float32)
                    mel_length_np = mel_length_tensor.cpu().numpy().astype(np.int64)
                    
                    yield {self.input_name: mel_np, self.length_name: mel_length_np}
                
                except Exception as e:
                    print(f"\nError processing file {audio_path}: {e}. Skipping.")
                    continue

def quantize_quartznet_model(model_path, output_path, dataset_name, max_samples=100): 
    print("=== Model Quantization ===")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Dataset: {dataset_name}") # In tên dataset HF
    print(f"Samples: {max_samples}")

    try:
        # Khởi tạo DataReader với tên dataset HF
        calibration_data_reader = MelSpecDataReader(model_path, dataset_name, max_samples, use_huggingface=True, split="test", max_duration=10.0)
        encoder_nodes_to_skip = [
            # ============================================
            # MỚI (PHẦN 1): LogSoftmax trong Decoder (Rất quan trọng)
            # ============================================
            '/decoder/LogSoftmax',
            '/decoder/LogSoftmax_output_0',

            # ============================================
            # Layer 0
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.0/self_attn/MatMul_1',
            '/encoder/layers.0/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.0/conv/Sigmoid',
            '/encoder/layers.0/conv/Sigmoid_output_0',
            '/encoder/layers.0/conv/Mul',
            '/encoder/layers.0/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.0/self_attn/Softmax',
            '/encoder/layers.0/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.0/feed_forward1/activation/Sigmoid',
            '/encoder/layers.0/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.0/feed_forward1/activation/Mul',
            '/encoder/layers.0/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.0/feed_forward2/activation/Sigmoid',
            '/encoder/layers.0/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.0/feed_forward2/activation/Mul',
            '/encoder/layers.0/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.0/conv/activation/Sigmoid',
            '/encoder/layers.0/conv/activation/Sigmoid_output_0',
            '/encoder/layers.0/conv/activation/Mul',
            '/encoder/layers.0/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.0/self_attn/MatMul',
            '/encoder/layers.0/self_attn/MatMul_output_0',
            '/encoder/layers.0/self_attn/MatMul_2',
            '/encoder/layers.0/self_attn/MatMul_2_output_0',
            '/encoder/layers.0/self_attn/Div',
            '/encoder/layers.0/self_attn/Div_output_0',

            # ============================================
            # Layer 1
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.1/self_attn/MatMul_1',
            '/encoder/layers.1/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.1/conv/Sigmoid',
            '/encoder/layers.1/conv/Sigmoid_output_0',
            '/encoder/layers.1/conv/Mul',
            '/encoder/layers.1/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.1/self_attn/Softmax',
            '/encoder/layers.1/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.1/feed_forward1/activation/Sigmoid',
            '/encoder/layers.1/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.1/feed_forward1/activation/Mul',
            '/encoder/layers.1/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.1/feed_forward2/activation/Sigmoid',
            '/encoder/layers.1/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.1/feed_forward2/activation/Mul',
            '/encoder/layers.1/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.1/conv/activation/Sigmoid',
            '/encoder/layers.1/conv/activation/Sigmoid_output_0',
            '/encoder/layers.1/conv/activation/Mul',
            '/encoder/layers.1/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.1/self_attn/MatMul',
            '/encoder/layers.1/self_attn/MatMul_output_0',
            '/encoder/layers.1/self_attn/MatMul_2',
            '/encoder/layers.1/self_attn/MatMul_2_output_0',
            '/encoder/layers.1/self_attn/Div',
            '/encoder/layers.1/self_attn/Div_output_0',

            # ============================================
            # Layer 2
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.2/self_attn/MatMul_1',
            '/encoder/layers.2/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.2/conv/Sigmoid',
            '/encoder/layers.2/conv/Sigmoid_output_0',
            '/encoder/layers.2/conv/Mul',
            '/encoder/layers.2/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.2/self_attn/Softmax',
            '/encoder/layers.2/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.2/feed_forward1/activation/Sigmoid',
            '/encoder/layers.2/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.2/feed_forward1/activation/Mul',
            '/encoder/layers.2/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.2/feed_forward2/activation/Sigmoid',
            '/encoder/layers.2/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.2/feed_forward2/activation/Mul',
            '/encoder/layers.2/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.2/conv/activation/Sigmoid',
            '/encoder/layers.2/conv/activation/Sigmoid_output_0',
            '/encoder/layers.2/conv/activation/Mul',
            '/encoder/layers.2/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.2/self_attn/MatMul',
            '/encoder/layers.2/self_attn/MatMul_output_0',
            '/encoder/layers.2/self_attn/MatMul_2',
            '/encoder/layers.2/self_attn/MatMul_2_output_0',
            '/encoder/layers.2/self_attn/Div',
            '/encoder/layers.2/self_attn/Div_output_0',

            # ============================================
            # Layer 3
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.3/self_attn/MatMul_1',
            '/encoder/layers.3/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.3/conv/Sigmoid',
            '/encoder/layers.3/conv/Sigmoid_output_0',
            '/encoder/layers.3/conv/Mul',
            '/encoder/layers.3/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.3/self_attn/Softmax',
            '/encoder/layers.3/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.3/feed_forward1/activation/Sigmoid',
            '/encoder/layers.3/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.3/feed_forward1/activation/Mul',
            '/encoder/layers.3/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.3/feed_forward2/activation/Sigmoid',
            '/encoder/layers.3/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.3/feed_forward2/activation/Mul',
            '/encoder/layers.3/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.3/conv/activation/Sigmoid',
            '/encoder/layers.3/conv/activation/Sigmoid_output_0',
            '/encoder/layers.3/conv/activation/Mul',
            '/encoder/layers.3/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.3/self_attn/MatMul',
            '/encoder/layers.3/self_attn/MatMul_output_0',
            '/encoder/layers.3/self_attn/MatMul_2',
            '/encoder/layers.3/self_attn/MatMul_2_output_0',
            '/encoder/layers.3/self_attn/Div',
            '/encoder/layers.3/self_attn/Div_output_0',

            # ============================================
            # Layer 4
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.4/self_attn/MatMul_1',
            '/encoder/layers.4/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.4/conv/Sigmoid',
            '/encoder/layers.4/conv/Sigmoid_output_0',
            '/encoder/layers.4/conv/Mul',
            '/encoder/layers.4/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.4/self_attn/Softmax',
            '/encoder/layers.4/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.4/feed_forward1/activation/Sigmoid',
            '/encoder/layers.4/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.4/feed_forward1/activation/Mul',
            '/encoder/layers.4/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.4/feed_forward2/activation/Sigmoid',
            '/encoder/layers.4/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.4/feed_forward2/activation/Mul',
            '/encoder/layers.4/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.4/conv/activation/Sigmoid',
            '/encoder/layers.4/conv/activation/Sigmoid_output_0',
            '/encoder/layers.4/conv/activation/Mul',
            '/encoder/layers.4/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.4/self_attn/MatMul',
            '/encoder/layers.4/self_attn/MatMul_output_0',
            '/encoder/layers.4/self_attn/MatMul_2',
            '/encoder/layers.4/self_attn/MatMul_2_output_0',
            '/encoder/layers.4/self_attn/Div',
            '/encoder/layers.4/self_attn/Div_output_0',

            # ============================================
            # Layer 5
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.5/self_attn/MatMul_1',
            '/encoder/layers.5/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.5/conv/Sigmoid',
            '/encoder/layers.5/conv/Sigmoid_output_0',
            '/encoder/layers.5/conv/Mul',
            '/encoder/layers.5/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.5/self_attn/Softmax',
            '/encoder/layers.5/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.5/feed_forward1/activation/Sigmoid',
            '/encoder/layers.5/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.5/feed_forward1/activation/Mul',
            '/encoder/layers.5/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.5/feed_forward2/activation/Sigmoid',
            '/encoder/layers.5/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.5/feed_forward2/activation/Mul',
            '/encoder/layers.5/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.5/conv/activation/Sigmoid',
            '/encoder/layers.5/conv/activation/Sigmoid_output_0',
            '/encoder/layers.5/conv/activation/Mul',
            '/encoder/layers.5/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.5/self_attn/MatMul',
            '/encoder/layers.5/self_attn/MatMul_output_0',
            '/encoder/layers.5/self_attn/MatMul_2',
            '/encoder/layers.5/self_attn/MatMul_2_output_0',
            '/encoder/layers.5/self_attn/Div',
            '/encoder/layers.5/self_attn/Div_output_0',

            # ============================================
            # Layer 6
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.6/self_attn/MatMul_1',
            '/encoder/layers.6/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.6/conv/Sigmoid',
            '/encoder/layers.6/conv/Sigmoid_output_0',
            '/encoder/layers.6/conv/Mul',
            '/encoder/layers.6/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.6/self_attn/Softmax',
            '/encoder/layers.6/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.6/feed_forward1/activation/Sigmoid',
            '/encoder/layers.6/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.6/feed_forward1/activation/Mul',
            '/encoder/layers.6/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.6/feed_forward2/activation/Sigmoid',
            '/encoder/layers.6/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.6/feed_forward2/activation/Mul',
            '/encoder/layers.6/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.6/conv/activation/Sigmoid',
            '/encoder/layers.6/conv/activation/Sigmoid_output_0',
            '/encoder/layers.6/conv/activation/Mul',
            '/encoder/layers.6/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.6/self_attn/MatMul',
            '/encoder/layers.6/self_attn/MatMul_output_0',
            '/encoder/layers.6/self_attn/MatMul_2',
            '/encoder/layers.6/self_attn/MatMul_2_output_0',
            '/encoder/layers.6/self_attn/Div',
            '/encoder/layers.6/self_attn/Div_output_0',

            # ============================================
            # Layer 7
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.7/self_attn/MatMul_1',
            '/encoder/layers.7/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.7/conv/Sigmoid',
            '/encoder/layers.7/conv/Sigmoid_output_0',
            '/encoder/layers.7/conv/Mul',
            '/encoder/layers.7/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.7/self_attn/Softmax',
            '/encoder/layers.7/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.7/feed_forward1/activation/Sigmoid',
            '/encoder/layers.7/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.7/feed_forward1/activation/Mul',
            '/encoder/layers.7/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.7/feed_forward2/activation/Sigmoid',
            '/encoder/layers.7/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.7/feed_forward2/activation/Mul',
            '/encoder/layers.7/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.7/conv/activation/Sigmoid',
            '/encoder/layers.7/conv/activation/Sigmoid_output_0',
            '/encoder/layers.7/conv/activation/Mul',
            '/encoder/layers.7/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.7/self_attn/MatMul',
            '/encoder/layers.7/self_attn/MatMul_output_0',
            '/encoder/layers.7/self_attn/MatMul_2',
            '/encoder/layers.7/self_attn/MatMul_2_output_0',
            '/encoder/layers.7/self_attn/Div',
            '/encoder/layers.7/self_attn/Div_output_0',

            # ============================================
            # Layer 8
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.8/self_attn/MatMul_1',
            '/encoder/layers.8/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.8/conv/Sigmoid',
            '/encoder/layers.8/conv/Sigmoid_output_0',
            '/encoder/layers.8/conv/Mul',
            '/encoder/layers.8/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.8/self_attn/Softmax',
            '/encoder/layers.8/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.8/feed_forward1/activation/Sigmoid',
            '/encoder/layers.8/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.8/feed_forward1/activation/Mul',
            '/encoder/layers.8/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.8/feed_forward2/activation/Sigmoid',
            '/encoder/layers.8/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.8/feed_forward2/activation/Mul',
            '/encoder/layers.8/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.8/conv/activation/Sigmoid',
            '/encoder/layers.8/conv/activation/Sigmoid_output_0',
            '/encoder/layers.8/conv/activation/Mul',
            '/encoder/layers.8/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.8/self_attn/MatMul',
            '/encoder/layers.8/self_attn/MatMul_output_0',
            '/encoder/layers.8/self_attn/MatMul_2',
            '/encoder/layers.8/self_attn/MatMul_2_output_0',
            '/encoder/layers.8/self_attn/Div',
            '/encoder/layers.8/self_attn/Div_output_0',

            # ============================================
            # Layer 9
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.9/self_attn/MatMul_1',
            '/encoder/layers.9/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.9/conv/Sigmoid',
            '/encoder/layers.9/conv/Sigmoid_output_0',
            '/encoder/layers.9/conv/Mul',
            '/encoder/layers.9/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.9/self_attn/Softmax',
            '/encoder/layers.9/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.9/feed_forward1/activation/Sigmoid',
            '/encoder/layers.9/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.9/feed_forward1/activation/Mul',
            '/encoder/layers.9/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.9/feed_forward2/activation/Sigmoid',
            '/encoder/layers.9/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.9/feed_forward2/activation/Mul',
            '/encoder/layers.9/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.9/conv/activation/Sigmoid',
            '/encoder/layers.9/conv/activation/Sigmoid_output_0',
            '/encoder/layers.9/conv/activation/Mul',
            '/encoder/layers.9/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.9/self_attn/MatMul',
            '/encoder/layers.9/self_attn/MatMul_output_0',
            '/encoder/layers.9/self_attn/MatMul_2',
            '/encoder/layers.9/self_attn/MatMul_2_output_0',
            '/encoder/layers.9/self_attn/Div',
            '/encoder/layers.9/self_attn/Div_output_0',

            # ============================================
            # Layer 10
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.10/self_attn/MatMul_1',
            '/encoder/layers.10/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.10/conv/Sigmoid',
            '/encoder/layers.10/conv/Sigmoid_output_0',
            '/encoder/layers.10/conv/Mul',
            '/encoder/layers.10/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.10/self_attn/Softmax',
            '/encoder/layers.10/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.10/feed_forward1/activation/Sigmoid',
            '/encoder/layers.10/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.10/feed_forward1/activation/Mul',
            '/encoder/layers.10/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.10/feed_forward2/activation/Sigmoid',
            '/encoder/layers.10/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.10/feed_forward2/activation/Mul',
            '/encoder/layers.10/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.10/conv/activation/Sigmoid',
            '/encoder/layers.10/conv/activation/Sigmoid_output_0',
            '/encoder/layers.10/conv/activation/Mul',
            '/encoder/layers.10/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.10/self_attn/MatMul',
            '/encoder/layers.10/self_attn/MatMul_output_0',
            '/encoder/layers.10/self_attn/MatMul_2',
            '/encoder/layers.10/self_attn/MatMul_2_output_0',
            '/encoder/layers.10/self_attn/Div',
            '/encoder/layers.10/self_attn/Div_output_0',

            # ============================================
            # Layer 11
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.11/self_attn/MatMul_1',
            '/encoder/layers.11/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.11/conv/Sigmoid',
            '/encoder/layers.11/conv/Sigmoid_output_0',
            '/encoder/layers.11/conv/Mul',
            '/encoder/layers.11/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.11/self_attn/Softmax',
            '/encoder/layers.11/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.11/feed_forward1/activation/Sigmoid',
            '/encoder/layers.11/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.11/feed_forward1/activation/Mul',
            '/encoder/layers.11/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.11/feed_forward2/activation/Sigmoid',
            '/encoder/layers.11/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.11/feed_forward2/activation/Mul',
            '/encoder/layers.11/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.11/conv/activation/Sigmoid',
            '/encoder/layers.11/conv/activation/Sigmoid_output_0',
            '/encoder/layers.11/conv/activation/Mul',
            '/encoder/layers.11/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.11/self_attn/MatMul',
            '/encoder/layers.11/self_attn/MatMul_output_0',
            '/encoder/layers.11/self_attn/MatMul_2',
            '/encoder/layers.11/self_attn/MatMul_2_output_0',
            '/encoder/layers.11/self_attn/Div',
            '/encoder/layers.11/self_attn/Div_output_0',

            # ============================================
            # Layer 12
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.12/self_attn/MatMul_1',
            '/encoder/layers.12/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.12/conv/Sigmoid',
            '/encoder/layers.12/conv/Sigmoid_output_0',
            '/encoder/layers.12/conv/Mul',
            '/encoder/layers.12/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.12/self_attn/Softmax',
            '/encoder/layers.12/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.12/feed_forward1/activation/Sigmoid',
            '/encoder/layers.12/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.12/feed_forward1/activation/Mul',
            '/encoder/layers.12/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.12/feed_forward2/activation/Sigmoid',
            '/encoder/layers.12/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.12/feed_forward2/activation/Mul',
            '/encoder/layers.12/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.12/conv/activation/Sigmoid',
            '/encoder/layers.12/conv/activation/Sigmoid_output_0',
            '/encoder/layers.12/conv/activation/Mul',
            '/encoder/layers.12/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.12/self_attn/MatMul',
            '/encoder/layers.12/self_attn/MatMul_output_0',
            '/encoder/layers.12/self_attn/MatMul_2',
            '/encoder/layers.12/self_attn/MatMul_2_output_0',
            '/encoder/layers.12/self_attn/Div',
            '/encoder/layers.12/self_attn/Div_output_0',

            # ============================================
            # Layer 13
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.13/self_attn/MatMul_1',
            '/encoder/layers.13/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.13/conv/Sigmoid',
            '/encoder/layers.13/conv/Sigmoid_output_0',
            '/encoder/layers.13/conv/Mul',
            '/encoder/layers.13/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.13/self_attn/Softmax',
            '/encoder/layers.13/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.13/feed_forward1/activation/Sigmoid',
            '/encoder/layers.13/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.13/feed_forward1/activation/Mul',
            '/encoder/layers.13/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.13/feed_forward2/activation/Sigmoid',
            '/encoder/layers.13/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.13/feed_forward2/activation/Mul',
            '/encoder/layers.13/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.13/conv/activation/Sigmoid',
            '/encoder/layers.13/conv/activation/Sigmoid_output_0',
            '/encoder/layers.13/conv/activation/Mul',
            '/encoder/layers.13/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.13/self_attn/MatMul',
            '/encoder/layers.13/self_attn/MatMul_output_0',
            '/encoder/layers.13/self_attn/MatMul_2',
            '/encoder/layers.13/self_attn/MatMul_2_output_0',
            '/encoder/layers.13/self_attn/Div',
            '/encoder/layers.13/self_attn/Div_output_0',

            # ============================================
            # Layer 14
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.14/self_attn/MatMul_1',
            '/encoder/layers.14/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.14/conv/Sigmoid',
            '/encoder/layers.14/conv/Sigmoid_output_0',
            '/encoder/layers.14/conv/Mul',
            '/encoder/layers.14/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.14/self_attn/Softmax',
            '/encoder/layers.14/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.14/feed_forward1/activation/Sigmoid',
            '/encoder/layers.14/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.14/feed_forward1/activation/Mul',
            '/encoder/layers.14/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.14/feed_forward2/activation/Sigmoid',
            '/encoder/layers.14/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.14/feed_forward2/activation/Mul',
            '/encoder/layers.14/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.14/conv/activation/Sigmoid',
            '/encoder/layers.14/conv/activation/Sigmoid_output_0',
            '/encoder/layers.14/conv/activation/Mul',
            '/encoder/layers.14/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.14/self_attn/MatMul',
            '/encoder/layers.14/self_attn/MatMul_output_0',
            '/encoder/layers.14/self_attn/MatMul_2',
            '/encoder/layers.14/self_attn/MatMul_2_output_0',
            '/encoder/layers.14/self_attn/Div',
            '/encoder/layers.14/self_attn/Div_output_0',

            # ============================================
            # Layer 15
            # ============================================
            # ĐÃ CÓ: MatMul_1 (Attention)
            '/encoder/layers.15/self_attn/MatMul_1',
            '/encoder/layers.15/self_attn/MatMul_1_output_0',
            # ĐÃ CÓ: GLU (Conv)
            '/encoder/layers.15/conv/Sigmoid',
            '/encoder/layers.15/conv/Sigmoid_output_0',
            '/encoder/layers.15/conv/Mul',
            '/encoder/layers.15/conv/Mul_output_0',
            # ĐÃ CÓ: Softmax (Attention)
            '/encoder/layers.15/self_attn/Softmax',
            '/encoder/layers.15/self_attn/Softmax_output_0',
            # ĐÃ CÓ: Swish (FFN 1)
            '/encoder/layers.15/feed_forward1/activation/Sigmoid',
            '/encoder/layers.15/feed_forward1/activation/Sigmoid_output_0',
            '/encoder/layers.15/feed_forward1/activation/Mul',
            '/encoder/layers.15/feed_forward1/activation/Mul_output_0',
            # ĐÃ CÓ: Swish (FFN 2)
            '/encoder/layers.15/feed_forward2/activation/Sigmoid',
            '/encoder/layers.15/feed_forward2/activation/Sigmoid_output_0',
            '/encoder/layers.15/feed_forward2/activation/Mul',
            '/encoder/layers.15/feed_forward2/activation/Mul_output_0',
            # MỚI (PHẦN 2): Swish (Conv)
            '/encoder/layers.15/conv/activation/Sigmoid',
            '/encoder/layers.15/conv/activation/Sigmoid_output_0',
            '/encoder/layers.15/conv/activation/Mul',
            '/encoder/layers.15/conv/activation/Mul_output_0',
            # MỚI (PHẦN 3): MatMul 0/2, Div (Attention)
            '/encoder/layers.15/self_attn/MatMul',
            '/encoder/layers.15/self_attn/MatMul_output_0',
            '/encoder/layers.15/self_attn/MatMul_2',
            '/encoder/layers.15/self_attn/MatMul_2_output_0',
            '/encoder/layers.15/self_attn/Div',
            '/encoder/layers.15/self_attn/Div_output_0',
        ]
        extra_options = {
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
            'AddQDQPairToWeight': True
         }
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=vai_q_onnx.QuantFormat.QOperator,
            calibrate_method=PowerOfTwoMethod.NonOverflow,
            nodes_to_exclude =  encoder_nodes_to_skip,
            extra_options = extra_options
        )
        print(f"--- Successfully quantized model saved to: {output_path} ---")

    except Exception as e:
        print(f"Quantization failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a combined Conformer CTC ONNX model using Hugging Face dataset.")
    parser.add_argument('--model', required=True, help='Path to the combined float ONNX model (encoder + decoder)')
    parser.add_argument('--output', required=True, help='Path to save the quantized ONNX model')
    # <<< THAY ĐỔI: Đổi tên đối số thành dataset_name >>>
    #parser.add_argument('--dataset_name', required=True, help='Hugging Face dataset name (e.g., linhtran92/viet_bud500)')
    parser.add_argument('--dataset_name',required= True,help = "Path to local dataset")
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to use for calibration (0 for all)')
    args = parser.parse_args()

    quantize_quartznet_model(
        model_path=args.model,
        output_path=args.output,
        dataset_name=args.dataset_name, # <<< Truyền tên dataset HF
        max_samples=args.samples
    )

