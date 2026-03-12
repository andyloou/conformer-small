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
import random
import math
import librosa
from loguru import logger
import torch.nn.functional as F
from vietasr.model import AudioToMelSpectrogramPreprocessor

class FilterbankFeatures(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_window_size=400,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=512,
        preemph=0.97,
        nfilt=80,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=0,
        frame_splicing=1,
        exact_pad=False, # Đảm bảo có tham số này
        pad_value=0,
        mag_power=2.0,
        rng=None, # Thêm
        nb_augmentation_prob=0.0, # Thêm
        nb_max_freq=4000, # Thêm
        mel_norm="slaney",
        stft_exact_pad=False, # Thêm
        stft_conv=False, # Thêm
    ):
        super().__init__()
        # Thêm logic của rng
        if rng is None:
            rng = random.Random()
        self.rng = rng

        if highfreq is None:
            highfreq = sample_rate / 2
        self.preemph = preemph
        self.n_fft = n_fft or n_window_size
        self.nfilt = nfilt
        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.pad_to = pad_to
        self.exact_pad = exact_pad # Đảm bảo có
        self.pad_value = pad_value
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.mag_power = mag_power
        # Thêm các thuộc tính mới
        self.nb_augmentation_prob = nb_augmentation_prob
        self.nb_max_freq = nb_max_freq
        self.mel_norm = mel_norm

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }
        # Thêm win_length, hop_length
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.register_buffer("window", torch_windows[window](n_window_size, periodic=False))
        
        # Thêm logic stft_pad_amount
        if exact_pad:
            self.stft_pad_amount = n_window_size // 2
        else:
            self.stft_pad_amount = None

        # Create mel filterbank
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=self.n_fft,
            n_mels=nfilt,
            fmin=lowfreq,
            fmax=highfreq,
            htk=False,
            norm=mel_norm if mel_norm else None # Cập nhật logic norm
        )
        mel_basis = mel_basis[None, :, :]  # Shape: [1, nfilt, n_fft//2 + 1]
        self.register_buffer("fb", torch.tensor(mel_basis).float())

    @torch.no_grad()
    def forward(self, audio, length):
        batch_size = audio.size(0) # Thêm
        if self.dither > 0:
            audio += self.dither * torch.randn_like(audio)
        
        # Cập nhật logic preemphasis
        if self.preemph is not None:
            preemph_audio = audio.new_zeros(audio.shape)
            preemph_audio[:, 1:] = audio[:, 1:] - self.preemph * audio[:, :-1]
            preemph_audio[:, 0] = audio[:, 0]
            audio = preemph_audio
            
        # *** LOGIC PADDING QUAN TRỌNG TỪ covert_to_onnx.py ***
        if self.exact_pad:
            pad_amount = self.stft_pad_amount
            audio = F.pad(audio.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)
            length += 2 * pad_amount
        else:
            pad_amount = (self.n_window_size - self.n_window_stride) // 2
            # Đảm bảo audio đủ dài cho ít nhất 1 frame
            if audio.size(1) < self.n_window_size:
                 pad_right = self.n_window_size - audio.size(1)
                 audio = F.pad(audio, (0, pad_right), mode="reflect")
            
            # Tính toán padding cần thiết
            # (Thêm điều kiện check audio.size(1) > pad_amount)
            needed_length = 0
            if audio.size(1) > pad_amount:
                 needed_length = pad_amount + math.ceil((audio.size(1) - pad_amount) / self.n_window_stride) * self.n_window_stride
            
            if needed_length > audio.size(1):
                pad_right = needed_length - audio.size(1)
                audio = F.pad(audio, (0, pad_right), mode="reflect")

        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.n_window_stride,
            win_length=self.n_window_size,
            window=self.window,
            center=False, # <-- THAY ĐỔI QUAN TRỌNG
            pad_mode='reflect', # (Mặc dù center=False, nhưng giữ lại)
            return_complex=True
        )
        
        # Compute power spectrogram
        mag = torch.abs(stft)
        power = mag ** self.mag_power  # Shape: [batch_size, n_fft//2 + 1, time]
        
        # Apply mel filterbank
        # Đảm bảo device matching
        mel = torch.matmul(self.fb.to(power.device), power)  
        
        # Log scale
        if self.log:
            if self.log_zero_guard_type == "add":
                mel = torch.log(mel + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                mel = torch.clamp(mel, min=self.log_zero_guard_value).log()
        
        # Normalize (Logic từ covert_to_onnx.py)
        if self.normalize == "per_feature":
            mean = mel.mean(dim=-1, keepdim=True)
            std = mel.std(dim=-1, keepdim=True) + 1e-5 # epsilon
            mel = (mel - mean) / std
        elif self.normalize == "all_features":
            mean = mel.mean(keepdim=True)
            std = mel.std(keepdim=True) + 1e-5
            mel = (mel - mean) / std
        
        # Frame splicing
        if self.frame_splicing > 1:
            mel = mel.reshape(mel.size(0), mel.size(1) // self.frame_splicing, mel.size(1) * self.frame_splicing)
        
        # Pad to
        if self.pad_to > 0:
            N = mel.size(-1)
            P = self.pad_to - N % self.pad_to
            if P > 0:
                mel = F.pad(mel, (0, P), value=self.pad_value)
                
        # Update length for STFT (Logic từ covert_to_onnx.py)
        length = (length - self.n_window_size) // self.n_window_stride + 1
        actual_time_steps = mel.shape[-1]
        length = torch.clamp(length, max=actual_time_steps)
        
        return mel, length




class MelSpecDataReader(CalibraterBase):
    def __init__(
        self,
        model_path,
        config: dict, # <<< THÊM MỚI: Nhận config
        max_samples=100,
        input_name='mel_spectrogram',
        length_name='mel_length',
        use_huggingface=True,
        dataset_name="linhtran92/viet_bud500",
        split="test",
        max_duration=10.0
    ):
        super().__init__(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("[MelSpecDataReader] Initializing Preprocessor for Calibration...")
        
        # --- THAY ĐỔI: Khởi tạo preprocessor chuẩn từ config ---
        preproc_cfg = config.get("preprocessor", {})
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preproc_cfg)
        self.preprocessor.to(self.device)
        self.preprocessor.eval()
        logger.info("[MelSpecDataReader] Preprocessor Initialized.")
        # --- Hết thay đổi ---

        self.use_huggingface = use_huggingface
        self.max_duration = max_duration
        self.target_sr = config["dataset"].get("target_sampling_rate", 16000)

        if use_huggingface:
            logger.info(f"[MelSpecDataReader] Loading HF dataset: {dataset_name} (split={split})")
            self.dataset = load_dataset(dataset_name, split=split, streaming=False)
            
            if max_duration:
                logger.info(f"[MelSpecDataReader] Filtering audios <= {max_duration}s")
                def filter_fn(example):
                    duration = len(example['audio']['array']) / example['audio']['sampling_rate']
                    return duration <= max_duration
                self.dataset = self.dataset.filter(filter_fn)
            
            total_samples = len(self.dataset)
            if max_samples > 0 and total_samples > max_samples:
                logger.info(f"[MelSpecDataReader] Limiting calibration to {max_samples} samples")
                indices = np.random.choice(total_samples, max_samples, replace=False)
                self.dataset = self.dataset.select(indices)
            
            logger.info(f"[MelSpecDataReader] Prepared {len(self.dataset)} samples.")
        else:
            # Logic load local (nếu bạn cần)
            raise NotImplementedError("Local dataset calibration not fully implemented in this refactor")

        self.input_name = input_name
        self.length_name = length_name
        self.batch_size = 1
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self._process_data())
        return next(self.enum_data, None)

    @torch.no_grad()
    def _process_data(self):
        if self.use_huggingface:
            for i, example in enumerate(self.dataset):
                try:
                    audio_dict = example['audio']
                    waveform_np = audio_dict['array']
                    sr = audio_dict['sampling_rate']
                    
                    waveform = torch.from_numpy(waveform_np).float().unsqueeze(0).to(self.device)
                    
                    if sr != self.target_sr:
                        waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
                    
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    length = torch.tensor([waveform.shape[1]], dtype=torch.long).to(self.device)
                    
                    # Chạy preprocessor chuẩn
                    mel_tensor, mel_length_tensor = self.preprocessor(waveform, length)
                    
                    mel_np = mel_tensor.cpu().numpy().astype(np.float32)
                    mel_length_np = mel_length_tensor.cpu().numpy().astype(np.int64)
                    
                    yield {self.input_name: mel_np, self.length_name: mel_length_np}
                
                except Exception as e:
                    logger.warning(f"\n[MelSpecDataReader] Error processing sample {i}: {e}. Skipping.")
                    continue
        else:
            pass # Bỏ qua logic file local cũ

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
    parser.add_argument('--output',default= "quan_conformer", help='Path to save the quantized ONNX model')
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

