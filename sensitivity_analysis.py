# sensitivity_analysis.py (NỘI DUNG MỚI HOÀN TOÀN)

import onnx
import onnxruntime as ort
import numpy as np
import torch
import torchaudio
from datasets import load_dataset
import json
from tqdm import tqdm
from vai_q_onnx import quantize_static, PowerOfTwoMethod
import os
from loguru import logger
from torch.utils.data import DataLoader
import vai_q_onnx # Thêm import

# === THÊM MỚI: Imports chuẩn từ vietasr và utils ===
try:
    from utils import load_config # Import từ file utils.py
    from vietasr.model import AudioToMelSpectrogramPreprocessor
    from vietasr.dataset.dataset import ASRDataset, ASRCollator
    from vietasr.utils.utils import calculate_wer # Import hàm WER chuẩn
except ImportError:
    logger.error("Không thể import 'vietasr' hoặc 'utils'.")
    logger.error("Vui lòng chạy script này trong môi trường đã cài 'vietasr' và có file 'utils.py'.")
    exit(1)
# ===================================================

class LayerSensitivityAnalyzer:
    """
    Phân tích độ nhạy cảm của từng layer/node với quantization
    """
    
    # === SỬA ĐỔI: __init__ ===
    def __init__(self, float_model_path, config: dict, test_dataset):
        logger.info("Initializing LayerSensitivityAnalyzer...")
        self.float_model_path = float_model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Setup Preprocessor (chuẩn)
        preproc_cfg = self.config.get("preprocessor", {})
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preproc_cfg)
        self.preprocessor.to(self.device)
        self.preprocessor.eval()

        # 2. Setup Collator (chuẩn)
        self.collator = ASRCollator(
            bpe_model_path=self.config["dataset"]["bpe_model_path"],
            target_sampling_rate=self.config["dataset"].get("target_sampling_rate", 16000)
        )
        self.vocab = self.collator.get_vocab()
        self.blank_id = len(self.vocab)
        text_key = 'transcription' 
        
        logger.info(f"Wrapping dataset. Using text key: '{text_key}'")
        # 'test_dataset' là raw HF dataset được truyền vào
        test_asr_dataset = ASRDataset(hf_dataset=test_dataset)
        
        self.dataloader = DataLoader(
            dataset=test_asr_dataset,
            batch_size=1, # Luôn dùng batch=1 để đánh giá
            num_workers=0,
            shuffle=False,
            collate_fn=self.collator
        )
        
        # 4. Load float session
        logger.info(f"Loading float model session: {float_model_path}")
        self.float_session = ort.InferenceSession(float_model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        
        # 5. Extract TẤT CẢ nodes (logic từ lần trước)
        self.all_nodes_map = self._extract_all_nodes_and_outputs()
        logger.info(f"Found {len(self.all_nodes_map)} total nodes in the graph")

    
    # (Hàm này đã sửa ở lần trước, giữ nguyên)
    def _extract_all_nodes_and_outputs(self):
        """Trích xuất TẤT CẢ nodes và output tensors của chúng"""
        model = onnx.load(self.float_model_path)
        nodes_and_tensors_map = {}
        for node in model.graph.node:
            names_to_skip = [node.name] + list(node.output)
            nodes_and_tensors_map[node.name] = names_to_skip
        return nodes_and_tensors_map

    # (Hàm này đã sửa ở lần trước, giữ nguyên)
    def _group_nodes_by_layer(self):
        """Nhóm TẤT CẢ nodes VÀ TENSORS theo layer"""
        layer_groups = {}
        for node_name, names_to_skip_list in self.all_nodes_map.items():
            layer_idx = None
            if '/encoder/layers.' in node_name:
                parts = node_name.split('/')
                for part in parts:
                    if part.startswith('layers.'):
                        layer_idx = part
                        break
            elif '/decoder/' in node_name:
                layer_idx = 'decoder'
            elif '/encoder/pre_encode/' in node_name:
                layer_idx = 'pre_encode'

            if layer_idx:
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = set()
                layer_groups[layer_idx].update(names_to_skip_list)
        
        # Chuyển set thành list
        final_groups = {idx: list(names) for idx, names in layer_groups.items()}
        return final_groups
    
    # === SỬA ĐỔI: quantize_except_nodes ===
    def quantize_except_nodes(self, nodes_to_skip, output_path):
        """Quantize model nhưng skip một số nodes"""
        
        # Import MelSpecDataReader đã được cập nhật từ quan.py
        from quan import MelSpecDataReader 
        
        logger.info(f"Quantizing to {output_path}, skipping {len(nodes_to_skip)} nodes/tensors...")

        # Khởi tạo DataReader VÀ TRUYỀN CONFIG VÀO
        calibration_data_reader = MelSpecDataReader(
            self.float_model_path,
            config=self.config, # <<< TRUYỀN CONFIG
            dataset_name="linhtran92/viet_bud500",
            max_samples=100,
            use_huggingface=True,
            split="test",
            max_duration=10.0
        )
        
        extra_options = {
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
            'AddQDQPairToWeight': True
        }
        
        quantize_static(
            model_input=self.float_model_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=vai_q_onnx.QuantFormat.QDQ, # <<< SỬ DỤNG QDQ
            calibrate_method=PowerOfTwoMethod.NonOverflow,
            nodes_to_exclude=nodes_to_skip,
            extra_options=extra_options
        )
        logger.info(f"Quantization complete: {output_path}")

    # === SỬA ĐỔI: evaluate_model (thay thế hoàn toàn) ===
    def evaluate_model(self, model_path, max_samples=None): # max_samples không còn dùng
        """Đánh giá WER của model (DÙNG DATALOADER CHUẨN)"""
        logger.info(f"Loading ONNX session for evaluation: {model_path}")
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
            logger.info(f"Session loaded on: {session.get_providers()[0]}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_path}: {e}")
            return 100.0

        all_predictions = []
        all_labels = []
        
        # Sử dụng self.dataloader đã được khởi tạo (chứa 100 samples)
        for batch in tqdm(self.dataloader, desc=f"Evaluating {os.path.basename(model_path)}"):
            try:
                # 1. Preprocess: audio -> mel (Dùng preprocessor chuẩn)
                audio = batch[0].to(self.device)
                audio_lens = batch[1].to(self.device)
                
                with torch.no_grad():
                    mel_feats, mel_lens = self.preprocessor(audio, audio_lens)
                
                # 2. ONNX Inference
                ort_inputs = {
                    'mel_spectrogram': mel_feats.cpu().numpy(),
                    'mel_length': mel_lens.cpu().numpy()
                }
                log_probs_onnx, encoded_length_onnx = session.run(None, ort_inputs)
                
                # 3. Decode ONNX (Dùng decoder và collator chuẩn)
                log_probs_onnx_np = log_probs_onnx[0, :encoded_length_onnx[0]]
                pred_ids = self._greedy_decode(log_probs_onnx_np)
                pred_text = self.collator.ids2text(pred_ids)
                all_predictions.append(pred_text)
                
                # 4. Get Ground truth (Dùng collator chuẩn)
                targets = batch[2]
                target_lens = batch[3]
                label_ids = targets[0, :target_lens[0]].tolist()
                label_text = self.collator.ids2text(label_ids)
                all_labels.append(label_text)
                
            except Exception as e:
                logger.warning(f"Error processing a batch: {e}. Skipping.")
                continue
        
        if not all_labels:
            logger.error("Evaluation failed: No samples were processed correctly.")
            return 100.0

        # 5. Calculate final WER (Dùng hàm calculate_wer chuẩn)
        logger.info(f"Calculating WER for {len(all_labels)} samples...")
        wer = calculate_wer(all_predictions, all_labels, use_cer=False)
        
        return wer # calculate_wer đã trả về %

    # === SỬA ĐỔI: _greedy_decode (giống test_model_goc_onnx.py) ===
    def _greedy_decode(self, log_probs):
        """CTC greedy decode"""
        argmax = np.argmax(log_probs, axis=1)
        result = []
        prev = self.blank_id
        
        for t in argmax:
            if t != self.blank_id and t != prev:
                result.append(int(t)) # Đảm bảo là int
            prev = t
        
        return result
    
    def analyze_layer_sensitivity(self, output_json='layer_sensitivity.json'):
        """
        Phân tích độ nhạy: Quantize từng layer và đo WER
        """
        print("=== LAYER SENSITIVITY ANALYSIS ===")
        
        # 1. Baseline: Float model WER
        print("\n1. Evaluating float model...")
        baseline_wer = self.evaluate_model(self.float_model_path)
        print(f"   Baseline WER: {baseline_wer:.2f}%")
        
        if baseline_wer >= 99.0:
            logger.error("Baseline WER is 100% or close. Analysis is meaningless.")
            logger.error(f"PLEASE CHECK YOUR TEXT KEY ('{self.dataloader.dataset.text_key}')")
            return {}

        # 2. Group nodes by layer
        layer_groups = self._group_nodes_by_layer()
        
        results = {
            'baseline_wer': baseline_wer,
            'layer_sensitivity': {}
        }
        
        # 3. Test từng layer
        for layer_name, nodes_to_skip in sorted(layer_groups.items()): # Thêm sorted()
            print(f"\n2. Testing {layer_name} (skipping {len(nodes_to_skip)} nodes/tensors)...")
            
            temp_model = f'temp_skip_{layer_name}.onnx'
            
            try:
                # Quantize TẤT CẢ NGOẠI TRỪ layer này
                self.quantize_except_nodes(nodes_to_skip, temp_model)
                
                # Đánh giá model tạm
                layer_wer = self.evaluate_model(temp_model)
                
                wer_increase = layer_wer - baseline_wer
                
                results['layer_sensitivity'][layer_name] = {
                    'wer': layer_wer,
                    'wer_increase': wer_increase,
                    'num_nodes': len(nodes_to_skip),
                }
                
                print(f"   WER when skipping {layer_name}: {layer_wer:.2f}% (+{wer_increase:.2f}%)")
                
            except Exception as e:
                logger.error(f"   Error processing {layer_name}: {e}")
                results['layer_sensitivity'][layer_name] = {'error': str(e)}
                continue
            finally:
                # Cleanup
                if os.path.exists(temp_model):
                    os.remove(temp_model)
        
        # 4. Save results
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Results saved to {output_json} ===")
        
        # 5. Recommend layers (ĐÃ SỬA LOGIC SẮP XẾP)
        self._recommend_skip_layers(results)
        
        return results
    
    # === SỬA ĐỔI: _recommend_skip_layers (Logic sắp xếp đúng) ===
    def _recommend_skip_layers(self, results):
        """Đề xuất layers nên skip dựa trên sensitivity"""
        print("\n=== RECOMMENDATIONS ===")
        
        sensitivities = []
        for layer_name, data in results['layer_sensitivity'].items():
            if 'wer_increase' in data: # Check nếu có kết quả hợp lệ
                sensitivities.append((layer_name, data['wer_increase']))
        
        if not sensitivities:
            print("No valid sensitivity data to analyze.")
            return

        # Sắp xếp theo wer_increase TĂNG DẦN (reverse=False)
        sensitivities.sort(key=lambda x: x[1], reverse=False) 
        
        print("\nMost sensitive layers (nên SKIP quantization):")
        # In ra các layer có wer_increase THẤP NHẤT
        for i, (layer, wer_inc) in enumerate(sensitivities[:5]):
            print(f"  {i+1}. {layer}: +{wer_inc:.2f}% WER increase (Gần baseline, rất nhạy cảm)")
        
        print("\nLeast sensitive layers (có thể quantize an toàn):")
        # Lấy 5 layer cuối cùng (có wer_increase cao nhất) và đảo ngược lại để in
        for i, (layer, wer_inc) in enumerate(reversed(sensitivities[-5:])):
            print(f"  {i+1}. {layer}: +{wer_inc:.2f}% WER increase (Ít nhạy cảm)")
    
    # (Hàm analyze_node_type_sensitivity chưa được sửa)
    def analyze_node_type_sensitivity(self, *args, **kwargs):
        print("\nNOTE: 'analyze_node_type_sensitivity' is not yet refactored.")
        return {}


# === SỬA ĐỔI: main() ===
def main():
    # Configuration
    FLOAT_MODEL = "conformer_vie.onnx"
    DATASET_NAME = "linhtran92/viet_bud500"
    CONFIG_PATH = "config/phase2.yaml" # <-- THÊM MỚI: Đường dẫn config
    
    # Load config
    logger.info(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    
    # Load test dataset (raw)
    logger.info("Loading test dataset...")
    test_ds = load_dataset(DATASET_NAME, split="test")
    # Giữ lại việc shuffle và select (tốt cho debug nhanh)
    test_ds = test_ds.shuffle(seed=42).select(range(100)) # Chỉ test 100 samples
    
    # Initialize analyzer (TRUYỀN CONFIG VÀO)
    analyzer = LayerSensitivityAnalyzer(FLOAT_MODEL, config, test_ds)
    
    # Run analysis
    print("\n" + "="*60)
    print("Running Layer-wise Analysis")
    print("="*60)
    layer_results = analyzer.analyze_layer_sensitivity()
    
    print("\n✅ Analysis complete! Check JSON files for detailed results.")


if __name__ == "__main__":
    main()