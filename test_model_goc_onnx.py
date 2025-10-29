import torch
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from vietasr.dataset.dataset import ASRDataset, ASRCollator
from vietasr.model import AudioToMelSpectrogramPreprocessor
from vietasr.utils.utils import calculate_wer
from utils import load_config
from vietasr.model import ConformerCTC
from datasets import load_from_disk
def test_onnx_model(
    onnx_path: str,
    config_path: str,
    pt_checkpoint_path: str = None,
    test_meta_filepath: str = None,
    use_huggingface: bool = True,
    dataset_name: str = "linhtran92/viet_bud500",
    device: str = "cuda",
    local_dataset_path = None
):
    """Test ONNX model và (tùy chọn) so sánh với PyTorch"""
    
    # Load config
    config = load_config(config_path)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Setup preprocessor (raw audio -> mel)
    preproc_cfg = config.get("preprocessor", {})
    preprocessor = AudioToMelSpectrogramPreprocessor(**preproc_cfg)
    preprocessor.to(device)
    preprocessor.eval()
    
    # Load PyTorch model (nếu được cung cấp)
    pt_model = None
    if pt_checkpoint_path:
        logger.info(f"Loading PyTorch model for comparison: {pt_checkpoint_path}")
        try:
            pt_model = ConformerCTC(vocab_path=config["dataset"]["bpe_model_path"], **config["model"])
            pt_model.load_checkpoint(pt_checkpoint_path, resume_mode="full")
            pt_model.to(device)
            pt_model.eval()
            logger.success("PyTorch model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            pt_model = None
    
    # Setup collator
    collator = ASRCollator(
        bpe_model_path=config["dataset"]["bpe_model_path"],
        target_sampling_rate=config["dataset"].get("target_sampling_rate", 16000)
    )
    vocab = collator.get_vocab()
    blank_id = len(vocab)
    
    # Load test dataset
    if use_huggingface:
        logger.info(f"Loading HuggingFace test set: {dataset_name}")
        test_dataset = ASRDataset(
            dataset_name=dataset_name,
            split="test",
            max_duration=config["dataset"].get("max_duration", 20.0)
        )

    else:
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

    
    # Load ONNX model
    logger.info(f"Loading ONNX model: {onnx_path}")
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    logger.success(f"ONNX model loaded on: {ort_session.get_providers()[0]}")
    
    # Test loop
    labels = []
    
    # --- CẬP NHẬT ---
    # Tách riêng list prediction cho 2 model
    onnx_predictions = [] 
    pt_predictions = [] 
    
    logger.info("Starting inference...")
    for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
        # Preprocess: audio -> mel
        audio = batch[0].to(device)
        audio_lens = batch[1].to(device)
        
        with torch.no_grad():
            mel_feats, mel_lens = preprocessor(audio, audio_lens)
        
        # --- 1. ONNX Inference ---
        ort_inputs = {
            'mel_spectrogram': mel_feats.cpu().numpy(),
            'mel_length': mel_lens.cpu().numpy()
        }
        log_probs_onnx, encoded_length_onnx = ort_session.run(None, ort_inputs)
        
        # Decode ONNX
        log_probs_onnx_np = log_probs_onnx[0, :encoded_length_onnx[0]]
        pred_ids_onnx = ctc_greedy_decode(log_probs_onnx_np, blank_id)
        pred_text_onnx = collator.ids2text(pred_ids_onnx)
        onnx_predictions.append(pred_text_onnx)
        
        # --- 2. PyTorch Inference (Optional) ---
        if pt_model:
            with torch.no_grad():
                pt_output = pt_model(mel_feats, mel_lens)
                # Lấy output từ Pytorch model
                log_probs_pt = pt_output["log_probs"] # [B, T, C]
                encoded_length_pt = pt_output["encoded_length"] # [B]
            
            # So sánh output thô (raw tensors)
            diff = np.abs(log_probs_onnx[0] - log_probs_pt[0].cpu().numpy()).max()
            if diff > 1e-3:
                logger.warning(f"Sample {i}: Large diff={diff:.6f}")

            # --- CẬP NHẬT: Decode PyTorch ---
            log_probs_pt_np = log_probs_pt[0, :encoded_length_pt[0]].cpu().numpy()
            pred_ids_pt = ctc_greedy_decode(log_probs_pt_np, blank_id)
            pred_text_pt = collator.ids2text(pred_ids_pt)
            pt_predictions.append(pred_text_pt)

        # --- 3. Ground truth ---
        targets = batch[2]
        target_lens = batch[3]
        label_ids = targets[0, :target_lens[0]].tolist()
        label_text = collator.ids2text(label_ids)
        labels.append(label_text)
        
        # Log mẫu (chỉ log ONNX cho gọn)
        if (i + 1) % 50 == 0:
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Label : {label_text}")
            logger.info(f"  ONNX  : {pred_text_onnx}")
            if pt_model:
                logger.info(f"  Pytorch: {pred_text_pt}")
    
    # --- CẬP NHẬT: TÍNH TOÁN VÀ IN KẾT QUẢ ---
    
    # 1. Tính toán ONNX
    wer_onnx = calculate_wer(onnx_predictions, labels, use_cer=False)
    cer_onnx = calculate_wer(onnx_predictions, labels, use_cer=True)
    
    # In kết quả ONNX
    logger.success("\n" + "="*60)
    logger.success("ONNX MODEL TEST RESULTS")
    logger.success("="*60)
    logger.success(f"Test samples: {len(onnx_predictions)}")
    logger.success(f"WER: {wer_onnx:.2f}%")
    logger.success(f"CER: {cer_onnx:.2f}%")
    logger.success("="*60)
    
    # 2. Tính toán và In kết quả PyTorch (nếu có)
    if pt_model:
        wer_pt = calculate_wer(pt_predictions, labels, use_cer=False)
        cer_pt = calculate_wer(pt_predictions, labels, use_cer=True)
        
        logger.info("\n" + "="*60)
        logger.info("PYTORCH MODEL TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Test samples: {len(pt_predictions)}")
        logger.info(f"WER: {wer_pt:.2f}%")
        logger.info(f"CER: {cer_pt:.2f}%")
        logger.info("="*60)

    # Trả về kết quả của ONNX (hoặc bạn có thể chọn trả về cả 2)
    return wer_onnx, cer_onnx

def ctc_greedy_decode(log_probs, blank_id):
    """Greedy CTC decode (giống model.py)"""
    argmax = np.argmax(log_probs, axis=1)
    prev = blank_id
    result = []
    for t in argmax:
        if t != blank_id and t != prev:
            result.append(int(t))
        prev = t
    return result

if __name__ == "__main__":
    # Đường dẫn
    ONNX_PATH = "conformer_vie.onnx"
    CONFIG_PATH = "config/phase2.yaml"
    
    # Đặt đường dẫn checkpoint PyTorch của bạn ở đây để so sánh
    # Đặt là None nếu chỉ muốn test ONNX
    PT_CHECKPOINT_PATH = "exps/bud500/conformer_phase2/epoch_37.pt" # Sửa lại đường dẫn này
    
    # Test với HuggingFace dataset
    test_onnx_model(
        onnx_path=ONNX_PATH,
        config_path=CONFIG_PATH,
        pt_checkpoint_path=PT_CHECKPOINT_PATH,
        use_huggingface=False,
        device="mps",
        local_dataset_path= "viet_bud500_processed"
    )