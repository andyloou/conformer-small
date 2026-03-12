
from typing import List, Tuple, Union
import io
import torch
import torchaudio
from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from vietasr.dataset.tokenizer import SentencepiecesTokenizer
from utils import pad_list
from datasets import load_dataset, Audio
from torch.utils.data import IterableDataset
import os
import numpy as np

class ASRDataset(IterableDataset):
    def __init__(self, dataset_name=None, split="train", max_duration=12.0, hf_dataset = None,cache_dir=None):
        self.max_duration = max_duration
        if hf_dataset is not None:
            self.dataset = hf_dataset.cast_column("audio",Audio(decode=False))
        else:
            self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            self.dataset = self.dataset.cast_column("audio",Audio(decode=False))

    def __iter__(self):
        for sample in self.dataset:
            audio_info = sample["audio"]
            text = sample["transcription"]

            # ƯU TIÊN 1: Dùng array đã có trong Arrow (load_from_disk)
            if "array" in audio_info and audio_info["array"] is not None:
                waveform = torch.from_numpy(np.array(audio_info["array"])).float()
                sample_rate = audio_info["sampling_rate"]
            else:
                # ƯU TIÊN 2: Dùng path (nếu có file thật)
                audio_path = audio_info.get("path")
                if audio_path and os.path.exists(audio_path):
                    waveform, sample_rate = torchaudio.load(audio_path)
                else:
                    # ƯU TIÊN 3: Dùng bytes (streaming)
                    audio_bytes = audio_info.get("bytes")
                    if audio_bytes:
                        buffer = io.BytesIO(audio_bytes)
                        waveform, sample_rate = torchaudio.load(buffer)
                    else:
                        logger.warning("No audio data (array/path/bytes) → skip")
                        continue

            # Chuyển về mono nếu cần
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(0)  # hoặc waveform[0]

            duration = waveform.shape[-1] / sample_rate
            if duration > self.max_duration:
                continue

            yield {
                "audio_array": waveform,
                "sample_rate": sample_rate,
                "text": text,
                "duration": duration,
            }
        
class ASRCollator():
    def __init__(
        self,
        bpe_model_path: str,
        target_sampling_rate: int = 16000  # Thêm tham số để resample nếu cần
    ):
        self.tokenizer = SentencepiecesTokenizer(bpe_model_path)
        vocab = self.tokenizer.get_vocab()
        vocab = vocab[3:]
        vocab = ["<blank>", "<unk>"] + vocab + ["<pad>"]
        self.vocab = vocab
        self.token2ids = {t:i for i,t in enumerate(vocab)}
        self.ids2token = {i:t for i,t in enumerate(vocab)}
        self.blank_id = 0
        self.unk_id = 1
        self.pad_id = len(vocab) - 1
        self.target_sampling_rate = target_sampling_rate
    
    def get_vocab(self):
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def text2ids(self, text: str):
        tokens = self.tokenizer.text2tokens(text)
        ids = [self.token2ids.get(t, self.unk_id) for t in tokens]
        return ids
        
    def ids2text(self, ids: List[int]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        tokens = [self.ids2token[i] for i in ids if i not in [self.blank_id, self.unk_id, self.pad_id]]
        text = self.tokenizer.tokens2text(tokens)
        return text
        
    def __call__(self, batch):
        inputs, input_lens, targets, target_lens = [], [], [], []

        for sample in batch:
            waveform = sample["audio_array"]  # đã là torch.Tensor
            sr = sample["sample_rate"]

            # Resample nếu cần
            if sr != self.target_sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.target_sampling_rate)
                waveform = resampler(waveform)

            # Đảm bảo 1D
            if waveform.dim() > 1:
                waveform = waveform.squeeze()

            inputs.append(waveform)
            input_lens.append(waveform.shape[0])

            target_ids = self.text2ids(sample["text"])
            targets.append(torch.LongTensor(target_ids))
            target_lens.append(len(target_ids))

        inputs = pad_list(inputs, 0.0)
        input_lens = torch.LongTensor(input_lens)
        targets = pad_list(targets, self.pad_id)
        target_lens = torch.LongTensor(target_lens)

        return inputs, input_lens, targets, target_lens
