import torch
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import os

from vietasr.dataset.dataset import ASRDataset, ASRCollator
from vietasr.model import AudioToMelSpectrogramPreprocessor
from vietasr.utils.utils import calculate_wer
from utils import load_config
from datasets import load_from_disk
from pyctcdecode import build_ctcdecoder

def test_onnx_model(
    onnx_path: str,
    config_path: str,
    device: str = "cuda",
    local_dataset_path = None
):
    config = load_config(config_path)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Setup preprocessor
    preproc_cfg = config.get("preprocessor", {})
    preprocessor = AudioToMelSpectrogramPreprocessor(**preproc_cfg)
    preprocessor.to(device)
    preprocessor.eval()
    
    # Setup collator
    collator = ASRCollator(
        bpe_model_path=config["dataset"]["bpe_model_path"],
        target_sampling_rate=config["dataset"].get("target_sampling_rate", 16000)
    )
    vocab = collator.get_vocab()
    blank_id = len(vocab)
    decode_cfg = config.get("decode", {})
    kenlm_path = decode_cfg.get("kenlm_path") 
    word_vocab_path = decode_cfg.get("word_vocab_path")

    predictor = None
    if not kenlm_path:
        logger.warning("Using greedy decoding as kenlm_path is not provided.")
    else:
        try:
            bpe_vocab = collator.get_vocab()
            labels = list(bpe_vocab)
            labels.append('')
            unigrams_list = None
            if word_vocab_path and os.path.exists(word_vocab_path):
                logger.info(f"Loading unigrams from: {word_vocab_path}")
                with open(word_vocab_path, 'r', encoding='utf-8') as f:
                    unigrams_list = [line.strip() for line in f]
                logger.success(f"Loaded {len(unigrams_list)} unigrams")
                logger.debug(f"First 5 unigrams: {unigrams_list[:5]}")
                
                # Kiểm tra overlap
                vocab_set = set(labels[:-1])  # Exclude blank
                unigram_set = set(unigrams_list)
                overlap = len(vocab_set & unigram_set)
                logger.info(f"Vocab-Unigram overlap: {overlap}/{len(vocab_set)} ({overlap/len(vocab_set)*100:.1f}%)")
            else:
                logger.warning(f"word_vocab_path not found: {word_vocab_path}")
            
            # 5. Verify KenLM file exists
            if not os.path.exists(kenlm_path):
                logger.error(f"KenLM file not found: {kenlm_path}")
                raise FileNotFoundError(f"KenLM file not found: {kenlm_path}")
            
            kenlm_size = os.path.getsize(kenlm_path) / (1024 * 1024)
            logger.info(f"KenLM file: {kenlm_path} ({kenlm_size:.2f} MB)")
            
            # 6. Build decoder
            alpha = decode_cfg.get("kenlm_alpha", 0.5)
            beta = decode_cfg.get("kenlm_beta", 1.5)
            predictor = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=kenlm_path,
                unigrams=unigrams_list,
                alpha=alpha,
                beta=beta
            )

        except Exception as e:
            logger.error(f"Failed to build decoder: {e}")
            logger.exception(e)
            predictor = None
            
    # Load dataset
    if local_dataset_path is None:
        local_dataset_path = config["dataset"]["local_dataset_path"]
    logger.info(f"Loading local dataset from: {local_dataset_path}")
    local_ds = load_from_disk(local_dataset_path)
    test_split = local_ds["test"]
    test_dataset = ASRDataset(hf_dataset=test_split)

    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=collator
    )
    
    logger.info(f"Loading ONNX model: {onnx_path}")
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    labels_list = []
    onnx_predictions = [] 
    
    logger.info("Starting inference...")
    for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
        audio = batch[0].to(device)
        audio_lens = batch[1].to(device)
        
        with torch.no_grad():
            mel_feats, mel_lens = preprocessor(audio, audio_lens)

        ort_inputs = {
            'mel_spectrogram': mel_feats.cpu().numpy(),
            'mel_length': mel_lens.cpu().numpy()
        }
        log_probs_onnx, encoded_length_onnx = ort_session.run(None, ort_inputs)

        if predictor:
            probs_onnx = np.exp(log_probs_onnx)
            probs_sample = probs_onnx[0, :encoded_length_onnx[0]]  
            assert probs_sample.shape[1] == len(labels), \
                f"Mismatch: probs shape {probs_sample.shape[1]} vs labels {len(labels)}"
            
            
            pred_text_onnx = predictor.decode(
                logits=probs_sample,  
                beam_width=decode_cfg.get("beam_size", 100)
            )
        else:
            log_probs_onnx_np = log_probs_onnx[0, :encoded_length_onnx[0]]
            pred_ids_onnx = ctc_greedy_decode(log_probs_onnx_np, blank_id)
            pred_text_onnx = collator.ids2text(pred_ids_onnx)
            
        onnx_predictions.append(pred_text_onnx)
        
        # Lấy ground truth
        targets = batch[2]
        target_lens = batch[3]
        label_ids = targets[0, :target_lens[0]].tolist()
        label_text = collator.ids2text(label_ids)
        labels_list.append(label_text)

        if (i + 1) % 50 == 0:
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Label : {label_text}")
            logger.info(f"  ONNX  : {pred_text_onnx}")
            
    # Tính toán WER/CER
    wer_onnx = calculate_wer(onnx_predictions, labels_list, use_cer=False)
    cer_onnx = calculate_wer(onnx_predictions, labels_list, use_cer=True)

    logger.success("ONNX MODEL TEST RESULTS (WITH LM)")
    logger.success(f"Test samples: {len(onnx_predictions)}")
    logger.success(f"WER: {wer_onnx:.2f}%")
    logger.success(f"CER: {cer_onnx:.2f}%")
    return wer_onnx, cer_onnx

def ctc_greedy_decode(log_probs, blank_id):
    """Greedy CTC decode (dự phòng)"""
    argmax = np.argmax(log_probs, axis=1)
    prev = blank_id
    result = []
    for t in argmax:
        if t != blank_id and t != prev:
            result.append(int(t))
        prev = t
    return result

if __name__ == "__main__":
    ONNX_PATH = "conformer_quantized.onnx"
    CONFIG_PATH = "config/phase2.yaml"
    
    test_onnx_model(
        onnx_path=ONNX_PATH,
        config_path=CONFIG_PATH,
        device="cuda",
        local_dataset_path= "/home/datasets/viet_bud500"
    )