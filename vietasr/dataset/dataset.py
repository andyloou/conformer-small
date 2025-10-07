
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

class ASRDataset(IterableDataset):
    def __init__(self, dataset_name="linhtran92/viet_bud500", split="train", max_duration=12.0):
        self.dataset_name = dataset_name
        self.split = split
        self.max_duration = max_duration

    def __iter__(self):
        hf_dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))

        for sample in hf_dataset:
            audio_info = sample["audio"]
            audio_path = audio_info.get("path", None)
            audio_bytes = audio_info.get("bytes", None)
            text = sample["transcription"]

            if audio_path is not None:
                # load từ file path
                waveform, sample_rate = torchaudio.load(audio_path)
            elif audio_bytes is not None:
                # load từ memory buffer
                buffer = io.BytesIO(audio_bytes)
                waveform, sample_rate = torchaudio.load(buffer)
            else:
                logger.warning("Audio sample không có path hoặc bytes, skip!")
                continue

            # Stereo → mono
            if waveform.shape[0] > 1:
                waveform = waveform[0, :]

            duration = waveform.shape[-1] / sample_rate
            if duration > self.max_duration:
                continue

            yield {
                "audio_array": waveform,
                "sample_rate": sample_rate,
                "text": text,
                "duration": duration
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
        
    def __call__(self, batch: List[dict]):
        inputs = []
        input_lens = []
        targets = []
        target_lens = []
        
        for sample in batch:
            # Lấy audio array từ sample
            audio_array = sample['audio_array']
            sampling_rate = sample['sample_rate']
            text = sample['text']
            
            # Convert numpy array sang tensor
            waveform = torch.FloatTensor(audio_array)
            
            # Xử lý stereo -> mono nếu cần
            if waveform.dim() > 1:
                if waveform.shape[0] == 2:  # Stereo
                    waveform = waveform[0]  # Lấy channel đầu tiên
                elif waveform.shape[1] == 2:  # Shape (L, 2)
                    waveform = waveform[:, 0]
            
            # Resample nếu sampling rate khác target
            if sampling_rate != self.target_sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=self.target_sampling_rate
                )
                waveform = resampler(waveform)
            
            # Đảm bảo waveform là 1D
            waveform = waveform.squeeze()
            
            inputs.append(waveform)
            input_lens.append(waveform.shape[0])

            target = torch.LongTensor(self.text2ids(text))
            targets.append(target)
            target_lens.append(target.shape[0])
        
        inputs = pad_list(inputs, pad_value=0.0)
        input_lens = torch.LongTensor(input_lens)
        
        targets = pad_list(targets, pad_value=self.pad_id)
        target_lens = torch.LongTensor(target_lens)
        
        return inputs, input_lens, targets, target_lens
