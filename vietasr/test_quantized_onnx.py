import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
import jiwer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from tqdm import tqdm
import math
import librosa
import torchaudio

# Import cho Vitis AI custom ops
from onnxruntime_extensions import get_library_path as _lib_path
# Import này có thể cần thiết tùy môi trường Vitis AI
# from vai_q_onnx.operators.vai_ops.qdq_ops import vai_dquantize 

# ===================================================================
# PHẦN 1: CÁC LỚP PREPROCESSOR (Giữ nguyên)
# ===================================================================
class FilterbankFeatures(torch.nn.Module):
    # ... (Code giữ nguyên) ...
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

# ===================================================================
# PHẦN 2: CÁC HÀM TIỆN ÍCH (Giữ nguyên)
# ===================================================================
def greedy_decoder_with_nemo_tokenizer(log_probs, tokenizer):
    # ... (Hàm này giữ nguyên) ...
    best_path = np.argmax(log_probs, axis=-1)
    merged_path = [best_path[0]] if len(best_path) > 0 else []
    for i in range(1, len(best_path)):
        if best_path[i] != best_path[i-1]: merged_path.append(best_path[i])
    blank_id = tokenizer.vocab_size
    decoded_ids = [p for p in merged_path if p != blank_id]
    return tokenizer.ids_to_text(np.array(decoded_ids, dtype=np.int32)).strip() if decoded_ids else ""

def load_librispeech_transcripts(dataset_path):
    # ... (Hàm này giữ nguyên) ...
    transcripts = {}
    files = glob.glob(os.path.join(dataset_path, "**", "*.trans.txt"), recursive=True)
    for file_path in files:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2: transcripts[parts[0]] = parts[1].lower()
    return transcripts

# ===================================================================
# PHẦN 3: SCRIPT ĐÁNH GIÁ CHÍNH
# ===================================================================

if __name__ == "__main__":
    # <<< Đường dẫn tới model ĐÃ LƯỢNG TỬ HÓA >>>
    QUANTIZED_MODEL_PATH = "quan_conformer_QO_NO.onnx" 
    
    NEMO_TOKENIZER_MODEL_PATH = "977d4e24975b431ebb44f2dfcdea8778_tokenizer.model"
    LIBRISPEECH_PATH = "LibriSpeech/test-clean"
    BATCH_SIZE = 16 

    # --- BƯỚC 1: KHỞI TẠO PREPROCESSOR (Giữ nguyên) ---
    print("Initializing exact NeMo preprocessor...")
    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=16000, window_size=0.025, window_stride=0.01,
        features=80, n_fft=512, dither=1.0e-05, preemph=0.97
    )
    preprocessor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor.to(device)
    print(f"Preprocessor loaded to {device}.")

    # --- BƯỚC 2: TẢI TOKENIZER VÀ MODEL LƯỢNG TỬ HÓA ---
    print(f"Loading NeMo tokenizer from {NEMO_TOKENIZER_MODEL_PATH}...")
    tokenizer = SentencePieceTokenizer(model_path=NEMO_TOKENIZER_MODEL_PATH)

    print(f"Loading Quantized ONNX model from {QUANTIZED_MODEL_PATH}...")
    so = ort.SessionOptions()

    # <<< ĐĂNG KÝ THƯ VIỆN CUSTOM OPS >>>
    #try:
        #so.register_custom_ops_library(_lib_path())
        #print("Successfully registered Vitis AI custom ops library.")
    #except Exception as e:
        #print(f"WARNING: Could not register Vitis AI custom ops library: {e}")
        #print("         Quantized model might fail.")
    
    # <<< CẤU HÌNH ĐA LÕI CPU (NẾU CHẠY TRÊN CPU) >>>
    if device.type == 'cpu':
        num_cores = os.cpu_count() or 1
        print(f"Configuring ONNX Runtime for multi-core CPU ({num_cores} cores)...")
        so.intra_op_num_threads = num_cores
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        so.inter_op_num_threads = 12 # Tùy chọn

    # <<< KHỞI TẠO SESSION KHÔNG CÓ `providers` >>>
    try:
        sess = ort.InferenceSession(QUANTIZED_MODEL_PATH, so, providers = ["CPUExecutionProvider"]) 
        print(f"Quantized model loaded successfully using: {sess.get_providers()}") 
    except Exception as e:
        print(f"ERROR loading quantized model: {e}")
        exit()

    # --- BƯỚC 3: CHUẨN BỊ DỮ LIỆU (Giữ nguyên) ---
    print(f"Loading transcripts from {LIBRISPEECH_PATH}...")
    transcripts_map = load_librispeech_transcripts(LIBRISPEECH_PATH)
    audio_files = sorted(glob.glob(os.path.join(LIBRISPEECH_PATH, "**", "*.flac"), recursive=True))
    
    valid_audio_files = []
    ground_truths_map = {}
    for f in audio_files:
        file_id = os.path.basename(f).replace('.flac', '')
        if file_id in transcripts_map:
            valid_audio_files.append(f)
            ground_truths_map[f] = transcripts_map[file_id]
    print(f"Found {len(valid_audio_files)} files with matching transcripts to evaluate.")

    # --- BƯỚC 4: VÒNG LẶP INFERENCE ---
    ground_truths = []
    predictions = []
    
    for i in tqdm(range(0, len(valid_audio_files), BATCH_SIZE), desc="Transcribing"):
        batch_files = valid_audio_files[i:i + BATCH_SIZE]
        batch_audio, batch_lengths = [], []
        current_ground_truths = [] 

        for filepath in batch_files:
            try:
                waveform, sr = torchaudio.load(filepath)
                if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                batch_audio.append(waveform.squeeze(0))
                batch_lengths.append(waveform.shape[1])
                current_ground_truths.append(ground_truths_map[filepath])
            except Exception as e:
                print(f"\nError loading audio {filepath}: {e}. Skipping.")
                continue 

        if not batch_audio: continue 

        padded_audio = torch.nn.utils.rnn.pad_sequence(batch_audio, batch_first=True, padding_value=0.0)
        lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long)
        
        padded_audio = padded_audio.to(device)
        lengths_tensor = lengths_tensor.to(device)

        mel_tensor, mel_lengths_tensor = preprocessor(input_signal=padded_audio, length=lengths_tensor)
        
        mel_np = mel_tensor.cpu().numpy()
        mel_lengths_np = mel_lengths_tensor.cpu().numpy().astype(np.int64)

        input_data = {
            "mel_spectrogram": mel_np,
            "mel_length": mel_lengths_np
        }
        try:
            # <<< LẤY CẢ HAI OUTPUT TỪ MODEL ONNX >>>
            results = sess.run(None, input_data)
            log_probs_batch = results[0] 
            # Output thứ hai LÀ độ dài sau encoder
            output_lengths_batch = results[1] 
            
            ground_truths.extend(current_ground_truths) 
            for j, log_prob in enumerate(log_probs_batch):
                # <<< SỬ DỤNG ĐỘ DÀI ĐÚNG TỪ OUTPUT CỦA MODEL >>>
                true_len_output_steps = output_lengths_batch[j] 
                
                # Cắt log_probs theo đúng độ dài này
                predicted_text = greedy_decoder_with_nemo_tokenizer(log_prob[:true_len_output_steps], tokenizer)
                predictions.append(predicted_text)
                
        except Exception as e:
            print(f"\nError during ONNX inference for batch starting at index {i}: {e}")
            print(f"Input shapes: mel={mel_np.shape}, len={mel_lengths_np.shape}")
            break 

    # --- BƯỚC 5: TÍNH TOÁN WER ---
    print("\n\nEvaluation finished. Calculating WER...")
    if not ground_truths or not predictions or len(ground_truths) != len(predictions):
         print(f"Error: Mismatch lengths GT={len(ground_truths)} vs HYP={len(predictions)}")
    else:
        print("\n--- Sample Predictions ---")
        for i in range(min(5, len(predictions))):
            print(f"GT : {ground_truths[i]}")
            print(f"HYP: {predictions[i]}")
            print("-" * 10)
        wer = jiwer.wer(ground_truths, predictions)
        print(f"=====================================")
        print(f"Quantized Model ONNX WER: {wer * 100:.2f}%") # Đổi tên cho rõ
        print(f"=====================================")
