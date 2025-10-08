import math
from loguru import logger
import random
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from abc import ABC, abstractmethod
import librosa  # Required for mel filterbank creation
from torch.nn import CTCLoss
INF_VAL = 1e4
import numpy as np

# Utility functions for weight initialization
def calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}.")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in if mode == 'fan_in' else fan_out

def tds_uniform_(tensor, mode='fan_in'):
    fan = calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)
    bound = std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def tds_normal_(tensor, mode='fan_in'):
    fan = calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

def init_weights(m, mode: Optional[str] = 'xavier_uniform'):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == 'tds_uniform':
                tds_uniform_(m.weight)
            elif mode == 'tds_normal':
                tds_normal_(m.weight)
            else:
                raise ValueError(f"Unknown Initialization mode: {mode}")
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

# Dummy context for autocast compatibility
class avoid_float16_autocast_context:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Minimal NeuralType stubs for compatibility
class NeuralType:
    def __init__(self, axes, elem_type=None):
        self.axes = axes
        self.elem_type = elem_type

class AudioSignal(NeuralType):
    def __init__(self, **kwargs):
        super().__init__(('B', 'T'), **kwargs)

class LengthsType(NeuralType):
    def __init__(self, **kwargs):
        super().__init__(('B',), **kwargs)

class MelSpectrogramType(NeuralType):
    def __init__(self, **kwargs):
        super().__init__(('B', 'D', 'T'), **kwargs)

class SpectrogramType(NeuralType):
    def __init__(self, **kwargs):
        super().__init__(('B', 'D', 'T'), **kwargs)

# Dummy typecheck decorator
def typecheck():
    def decorator(func):
        return func
    return decorator

# FilterbankFeatures implementation
class FilterbankFeatures(nn.Module):
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
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,
        stft_conv=False,
    ):
        super().__init__()
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
        self.exact_pad = exact_pad
        self.pad_value = pad_value
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value = log_zero_guard_value
        self.mag_power = mag_power
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
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.register_buffer("window", torch_windows[window](n_window_size, periodic=False))
        if exact_pad:
            self.stft_pad_amount = n_window_size // 2
        else:
            self.stft_pad_amount = None
        # Create mel filterbank with shape [1, nfilt, n_fft//2 + 1]
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=self.n_fft,
            n_mels=nfilt,
            fmin=lowfreq,
            fmax=highfreq,
            htk=False,
            norm=mel_norm if mel_norm else None
        )
        mel_basis = mel_basis[None, :, :]  # Shape: [1, nfilt, n_fft//2 + 1]
        self.register_buffer("fb", torch.tensor(mel_basis).float())

    @torch.no_grad()
    def forward(self, audio, length):
        batch_size = audio.size(0)
        if self.dither > 0:
            audio += self.dither * torch.randn_like(audio)
        if self.preemph is not None:
            preemph_audio = audio.new_zeros(audio.shape)
            preemph_audio[:, 1:] = audio[:, 1:] - self.preemph * audio[:, :-1]
            preemph_audio[:, 0] = audio[:, 0]
            audio = preemph_audio
        if self.exact_pad:
            pad_amount = self.stft_pad_amount
            audio = F.pad(audio.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)
            length += 2 * pad_amount
        else:
            pad_amount = (self.n_window_size - self.n_window_stride) // 2
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
            center=False,
            pad_mode='reflect',
            return_complex=True
        )
        # Compute power spectrogram
        mag = torch.abs(stft)
        power = mag ** self.mag_power  # Shape: [batch_size, n_fft//2 + 1, time]
        # Apply mel filterbank
        mel = torch.matmul(self.fb, power)  # [1, nfilt, n_fft//2 + 1] x [batch_size, n_fft//2 + 1, time] -> [batch_size, nfilt, time]
        # Log scale
        if self.log:
            if self.log_zero_guard_type == "add":
                mel = torch.log(mel + self.log_zero_guard_value)
            elif self.log_zero_guard_type == "clamp":
                mel = torch.clamp(mel, min=self.log_zero_guard_value).log()
        # Normalize
        if self.normalize == "per_feature":
            mean = mel.mean(dim=-1, keepdim=True)
            std = mel.std(dim=-1, keepdim=True) + 1e-5
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
        # Update length for STFT
        length = (length - self.n_window_size) // self.n_window_stride + 1
        actual_time_steps = mel.shape[-1]
        length = torch.clamp(length, max=actual_time_steps)
        return mel, length

# AudioToMelSpectrogramPreprocessor
class AudioToMelSpectrogramPreprocessor(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        normalize="per_feature",
        n_fft=512,
        log=True,
        frame_splicing=1,
        dither=1.0e-5,
        pad_to=0,
        pad_value=0.0,
        features=80,
        lowfreq=0,
        highfreq=None,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        mag_power=2.0,
        preemph=0.97,
        exact_pad=False,
        mel_norm="slaney",
    ):
        super().__init__()
        n_window_size = int(window_size * sample_rate)
        n_window_stride = int(window_stride * sample_rate)
        self._sample_rate = sample_rate
        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            mel_norm=mel_norm,
        )
        self.register_buffer("dtype_sentinel_tensor", torch.tensor((), dtype=torch.float32), persistent=False)

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        if input_signal.dtype != torch.float32:
            input_signal = input_signal.to(torch.float32)
        processed_signal, processed_length = self.featurizer(input_signal, length)
        processed_signal = processed_signal.to(self.dtype_sentinel_tensor.dtype)
        return processed_signal, processed_length

    @property
    def filter_banks(self):
        return self.featurizer.fb

# SpecAugment
class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, time_masks=5, freq_width=27, time_width=0.05, rng=None, mask_value=0.0, use_vectorized_code=True):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.mask_value = mask_value
        self.use_vectorized_code = use_vectorized_code
        if rng is None:
            rng = random.Random()
        self.rng = rng

    @torch.no_grad()
    def forward(self, input_spec, length=None):
        B, F, T = input_spec.size()
        if self.use_vectorized_code:
            if self.freq_masks > 0:
                for idx in range(B):
                    for _ in range(self.freq_masks):
                        x = self.rng.randint(0, F - self.freq_width)
                        input_spec[idx, x : x + self.freq_width, :] = self.mask_value
            if self.time_masks > 0:
                if length is None:
                    length = input_spec.new_full((B,), T, dtype=torch.int64)
                time_width = self.time_width
                if isinstance(time_width, float):
                    time_width = int(time_width * length.max().item())
                for idx in range(B):
                    l = length[idx].item()
                    for _ in range(self.time_masks):
                        x = self.rng.randint(0, l - time_width)
                        input_spec[idx, :, x : x + time_width] = self.mask_value
        else:
            if self.freq_masks > 0:
                for _ in range(self.freq_masks):
                    x = self.rng.randint(0, F - self.freq_width)
                    input_spec[:, x : x + self.freq_width, :] = self.mask_value
            if self.time_masks > 0:
                if length is None:
                    num_cols = T
                else:
                    num_cols = length.min().item()
                for _ in range(self.time_masks):
                    x = self.rng.randint(0, num_cols - self.time_width)
                    input_spec[:, :, x : x + self.time_width] = self.mask_value
        return input_spec

# SpectrogramAugmentation
class SpectrogramAugmentation(nn.Module):
    def __init__(
        self,
        freq_masks=2,
        time_masks=5,
        freq_width=27,
        time_width=0.05,
        rect_masks=0,
        rect_time=5,
        rect_freq=20,
        rng=None,
        mask_value=0.0,
        use_vectorized_spec_augment=True,
        use_numba_spec_augment=False,
    ):
        super().__init__()
        if rect_masks > 0:
            self.spec_cutout = SpecCutout(
                rect_masks=rect_masks,
                rect_time=rect_time,
                rect_freq=rect_freq,
                rng=rng,
            )
        else:
            self.spec_cutout = lambda x: x
        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
                use_vectorized_code=use_vectorized_spec_augment,
            )
        else:
            self.spec_augment = lambda x, length=None: x
        self.spec_augment_numba = None

    @typecheck()
    def forward(self, input_spec, length):
        augmented_spec = self.spec_cutout(input_spec)
        if self.spec_augment_numba is not None and spec_augment_launch_heuristics(augmented_spec, length):
            augmented_spec = self.spec_augment_numba(input_spec=augmented_spec, length=length)
        else:
            augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)
        return augmented_spec

# SpecCutout
class SpecCutout(nn.Module):
    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super().__init__()
        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq
        if rng is None:
            rng = random.Random()
        self.rng = rng

    @torch.no_grad()
    def forward(self, input_spec):
        sh = input_spec.shape
        for _ in range(self.rect_masks):
            t0 = self.rng.randint(0, sh[2] - self.rect_time)
            t1 = t0 + self.rect_time
            f0 = self.rng.randint(0, sh[1] - self.rect_freq)
            f1 = f0 + self.rect_freq
            input_spec[:, f0:f1, t0:t1] = 0
        return input_spec

# spec_augment_launch_heuristics
def spec_augment_launch_heuristics(input_spec, length):
    return input_spec.is_cuda and length is not None

# Conformer-specific classes (unchanged from previous)
class ConvSubsampling(nn.Module):
    def __init__(
        self,
        subsampling='striding',
        subsampling_factor=4,
        feat_in=80,
        feat_out=176,
        conv_channels=176,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        super().__init__()
        if subsampling != 'striding':
            raise ValueError(f"This simplified version only supports 'striding', got: {subsampling}")
        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiple of 2!")
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2
        layers = []
        in_channels = 1
        for i in range(self._sampling_num):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                )
            )
            layers.append(activation)
            in_channels = conv_channels
        self.conv = nn.Sequential(*layers)
        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = self.calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = nn.Linear(conv_channels * int(out_length), feat_out)

    def calc_length(self, lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
        add_pad: float = all_paddings - kernel_size
        one: float = 1.0
        for i in range(repeat_num):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
            if ceil_mode:
                lengths = torch.ceil(lengths)
            else:
                lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)

    def forward(self, x, lengths):
        out_lengths = self.calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        x = x.unsqueeze(1)  # [B, 1, T, feat_in]
        x = self.conv(x)  # [B, conv_channels, T', F']
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, -1)  # [B, T', C*F']
        x = self.out(x)  # [B, T', feat_out]
        return x, out_lengths

class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        self.dropout_emb = nn.Dropout(dropout_rate_emb) if dropout_rate_emb > 0 else None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        if self.xscale:
            x = x * self.xscale
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, use_bias=True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = Swish()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size, norm_type='batch_norm', use_bias=True):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1, padding=padding, groups=d_model, bias=use_bias)
        self.batch_norm = nn.BatchNorm1d(d_model) if norm_type == 'batch_norm' else nn.LayerNorm(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=use_bias)

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        if isinstance(self.batch_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x

class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u=None, pos_bias_v=None, max_cache_len=0, use_bias=True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        return q, k, v

    def rel_shift(self, x):
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)
            n_batch_pos = pos_emb.size(0)
            n_batch = value.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)
            matrix_bd = matrix_bd[:, :, :, :matrix_ac.size(-1)]
            scores = (matrix_ac + matrix_bd) / self.s_d_k   
            return self.forward_attention(v, scores, mask)

class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_att=0.1,
        att_context_size=[-1, -1],
        use_bias=True,
    ):
        super().__init__()
        if self_attention_model != 'rel_pos':
            raise ValueError("This simplified version only supports 'rel_pos' attention model")
        self.self_attention_model = self_attention_model
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model=d_model, kernel_size=conv_kernel_size, norm_type=conv_norm_type, use_bias=use_bias)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(
            n_head=n_heads,
            n_feat=d_model,
            dropout_rate=dropout_att,
            max_cache_len=att_context_size[0],
            use_bias=use_bias,
        )
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_self_att(residual)
        x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        residual = residual + self.dropout(x)
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x)
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_out(residual)
        return x

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        feat_in=80,
        n_layers=16,
        d_model=176,
        feat_out=-1,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=176,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=[-1, -1],
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        dropout=0.1,
        dropout_emb=0.0,
        dropout_att=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.subsampling_factor = subsampling_factor
        self.att_context_size = att_context_size
        self.pos_emb_max_len = pos_emb_max_len
        d_ff = d_model * ff_expansion_factor
        self.xscale = math.sqrt(d_model) if xscaling else None
        self.pre_encode = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            subsampling_conv_chunking_factor=1,
            activation=nn.ReLU(True),
            is_causal=False,
        )
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout,
            max_len=pos_emb_max_len,
            xscale=self.xscale,
            dropout_rate_emb=dropout_emb,
        )
        self.layers = nn.ModuleList([
            ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type='batch_norm',
                dropout=dropout,
                dropout_att=dropout_att,
                att_context_size=att_context_size,
                use_bias=True,
            ) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, feat_out) if feat_out > 0 and feat_out != d_model else None
        self._feat_out = feat_out if feat_out > 0 else d_model
        self.max_audio_length = pos_emb_max_len
        self.set_max_audio_length(pos_emb_max_len)
        self.apply(lambda x: init_weights(x, mode='xavier_uniform'))

    def set_max_audio_length(self, max_audio_length):
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def update_max_seq_length(self, seq_length: int, device):
        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def _create_masks(self, att_context_size, padding_length, max_audio_length, device):
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(padding_length.size(0), -1) < padding_length.unsqueeze(-1)
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
        att_mask = att_mask[:, :max_audio_length, :max_audio_length]
        att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
        att_mask = ~att_mask
        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def forward(self, audio_signal, length):
        if length is None:
            length = audio_signal.new_full((audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64)
        self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        audio_signal = torch.transpose(audio_signal, 1, 2) if audio_signal.dim() == 3 and audio_signal.size(1) == self._feat_in else audio_signal
        audio_signal, length = self.pre_encode(audio_signal, length)
        length = length.to(torch.int64)
        max_audio_length = audio_signal.size(1)
        audio_signal, pos_emb = self.pos_enc(audio_signal)
        pad_mask, att_mask = self._create_masks(self.att_context_size, length, max_audio_length, audio_signal.device)
        for layer in self.layers:
            audio_signal = layer(audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)
        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length

class ConvASRDecoder(nn.Module):
    def __init__(self, feat_in=176, num_classes=1024, init_mode="xavier_uniform", vocabulary=None, add_blank=True):
        super().__init__()
        if vocabulary is None and num_classes < 0:
            raise ValueError("Neither vocabulary nor num_classes are set!")
        if vocabulary is not None:
            if num_classes > 0 and num_classes != len(vocabulary):
                raise ValueError(f"Vocabulary length must equal num_classes: {num_classes} vs {len(vocabulary)}")
            num_classes = len(vocabulary)
            self.vocabulary = vocabulary
        else:
            self.vocabulary = None
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self._feat_in = feat_in
        self._num_classes = num_classes + 1 if add_blank else num_classes
        self.decoder_layers = nn.Sequential(nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True))
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.temperature = 1.0

    def forward(self, encoder_output):
        logits = self.decoder_layers(encoder_output)
        logits = logits.transpose(1, 2)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return F.log_softmax(logits, dim=-1)

    @property
    def num_classes_with_blank(self):
        return self._num_classes


# ConformerCTC model
class ConformerCTC(nn.Module):
    def __init__(self, vocab_size, pad_id, **model_config):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.blank_id = vocab_size  # Blank là class cuối sau add_blank=True

        # Bỏ preprocessor parse/init

        # Giữ spec_augmentation
        spec_aug_cfg = {
            'freq_masks': model_config.get('freq_masks', 2),
            'time_masks': model_config.get('time_masks', 5),
            'freq_width': model_config.get('freq_width', 27),
            'time_width': model_config.get('time_width', 0.05),
            'rect_masks': 0,  # Không dùng
        }
        self.spec_augmentation = SpectrogramAugmentation(**spec_aug_cfg)

        # Encoder parse giống cũ
        encoder_params = model_config.get('encoder_params', {})
        encoder_cfg = {
            'feat_in': model_config.get('num_mel_bins', 80),
            'n_layers': encoder_params.get('n_layers', 16),
            'd_model': encoder_params.get('d_model', 176),
            'feat_out': -1,
            'subsampling': model_config.get('subsampling', 'striding'),
            'subsampling_factor': model_config.get('subsampling_factor', 4),
            'subsampling_conv_channels': model_config.get('subsampling_channel', 176),
            'ff_expansion_factor': encoder_params.get('ffn_mult', 4),
            'self_attention_model': model_config.get('self_attention_model', 'rel_pos'),
            'n_heads': encoder_params.get('nhead', 4),
            'att_context_size': model_config.get('att_context_size', [-1, -1]),
            'xscaling': model_config.get('xscaling', True),
            'untie_biases': model_config.get('untie_biases', True),
            'pos_emb_max_len': model_config.get('pos_emb_max_len', 5000),
            'conv_kernel_size': encoder_params.get('conv_kernel_size', 31),
            'dropout': encoder_params.get('dropout', 0.1),
            'dropout_emb': model_config.get('dropout_emb', 0.0),
            'dropout_att': encoder_params.get('dropout_att', 0.1),
        }
        self.encoder = ConformerEncoder(**encoder_cfg)

        # Decoder giống cũ
        d_model = encoder_cfg['d_model']
        self.decoder = ConvASRDecoder(
            feat_in=d_model,
            num_classes=vocab_size,
            vocabulary=None,
            add_blank=True,
        )

        # CTC Loss giống cũ
        self.ctc_loss = CTCLoss(
            reduction='mean',
            zero_infinity=True,
            blank=self.blank_id,
        )

    def forward(self, mel_feats, mel_lens, targets=None, target_lens=None):
        # mel_feats: [B, D=80, T], mel_lens: [B]

        # Augment nếu training
        if self.training:
            mel_feats = self.spec_augmentation(mel_feats, mel_lens)

        # Encode
        encoded, enc_lens = self.encoder(mel_feats, mel_lens)

        # Decode to log_probs
        log_probs = self.decoder(encoded)

        output = {
            "encoder_out": encoded,
            "encoder_out_lens": enc_lens,
            "log_probs": log_probs,
        }

        # Loss nếu có targets
        if targets is not None and target_lens is not None:
            log_probs_t = log_probs.transpose(0, 1)
            loss = self.ctc_loss(
                log_probs_t, targets, enc_lens, target_lens
            )
            output.update({
                "loss": loss,
                "ctc_loss": loss,
                "decoder_loss": torch.tensor(0.0, device=loss.device),
            })

        return output

    def forward_encoder(self, mel_feats, mel_lens):
        """Chỉ forward đến encoder"""
        encoded, enc_lens = self.encoder(mel_feats, mel_lens)
        return encoded, enc_lens

    def get_predicts(self, encoder_out, encoder_out_lens):
        """Greedy CTC decode"""
        log_probs = self.decoder(encoder_out)
        batch_size = log_probs.size(0)
        predicts = []
        for i in range(batch_size):
            lp = log_probs[i, :encoder_out_lens[i]].cpu().detach().numpy()
            pred_ids = ctc_greedy_decode(lp, self.blank_id)
            predicts.append(torch.tensor(pred_ids, dtype=torch.long))
        return predicts

    def get_labels(self, targets, target_lens):
        """Trích labels theo lens"""
        batch_size = targets.size(0)
        labels = []
        for i in range(batch_size):
            labels.append(targets[i, :target_lens[i]])
        return labels
    def load_checkpoint(self, checkpoint_path: str, resume_mode: str = "selective"):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # FLEXIBLE: Handle 'model', 'state_dict', hoặc raw dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            # Nếu có optimizer/lr_scheduler (resume), extract riêng (return để task dùng)
            optimizer_state = checkpoint.get('optimizer', None)
            lr_scheduler_state = checkpoint.get('lr_scheduler', None)
            epoch = checkpoint.get('epoch', None)
        else:
            state_dict = checkpoint  # Raw weights
            optimizer_state = lr_scheduler_state = epoch = None

        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid checkpoint structure at {checkpoint_path}: No 'model' or 'state_dict' found.")

        model_state_dict = self.state_dict()
        matched_state_dict = {}
        skipped_keys = []

        if resume_mode == "selective":
            # Selective: Chỉ load encoder (an toàn cho pretrained English)
            logger.info("Selective load: Chỉ load encoder weights")
            for key in state_dict.keys():
                if key.startswith('encoder.'):
                    if key in model_state_dict:
                        if state_dict[key].shape == model_state_dict[key].shape:
                            matched_state_dict[key] = state_dict[key]
                        else:
                            skipped_keys.append(f"{key} (shape mismatch)")
                    else:
                        skipped_keys.append(f"{key} (not in model)")
                else:
                    skipped_keys.append(key)  # Bỏ decoder, spec_aug, etc.

            # Warn nếu encoder key missing
            for key in model_state_dict.keys():
                if key.startswith('encoder.') and key not in matched_state_dict:
                    logger.warning(f"Encoder key {key} not found in checkpoint (random init)")
        else:
            # Full: Load tất cả (cho pretrained Vietnamese hoặc resume)
            logger.info("Full load: Load cả encoder và decoder weights")
            for key in state_dict.keys():
                if key in model_state_dict:
                    if state_dict[key].shape == model_state_dict[key].shape:
                        matched_state_dict[key] = state_dict[key]
                    else:
                        skipped_keys.append(f"{key} (shape mismatch)")
                else:
                    skipped_keys.append(f"{key} (not in model)")

            # Warn nếu key nào trong model không có trong checkpoint
            for key in model_state_dict.keys():
                if key not in matched_state_dict:
                    logger.warning(f"Model key {key} not found in checkpoint (random init)")

        model_state_dict.update(matched_state_dict)
        self.load_state_dict(model_state_dict)

        logger.info(f"Loaded checkpoint from {checkpoint_path}. Matched {len(matched_state_dict)} keys (mode: {resume_mode}).")
        if skipped_keys:
            logger.info(f"Skipped: {len(skipped_keys)} keys (e.g., {skipped_keys[:5]})")
        if resume_mode == "selective":
            logger.info("Decoder remains random (for Vietnamese vocab).")

        # Return extras cho resume
        return optimizer_state, lr_scheduler_state, epoch

def ctc_greedy_decode(log_probs, blank_id):
    """Greedy CTC decode: argmax + remove duplicates/blank"""
    argmax = np.argmax(log_probs, axis=1)
    prev = blank_id
    result = []
    for t in argmax:
        if t != blank_id and t != prev:
            result.append(t)
        prev = t
    return result

