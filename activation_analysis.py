import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from datasets import load_dataset
import torch
from loguru import logger
from utils import load_config
import torchaudio
from vietasr.model import AudioToMelSpectrogramPreprocessor
class ActivationAnalyzer:
    """
    Phân tích distribution của activations để xác định nodes nhạy cảm với quantization
    
    Nguyên lý:
    - Nodes có activation range lớn → khó quantize
    - Nodes có outliers → mất nhiều info khi quantize
    - Nodes có distribution không uniform → quantization error cao
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        
        # Extract intermediate outputs
        self.intermediate_names = self._get_all_intermediate_tensors()
        
    def _get_all_intermediate_tensors(self):
        """Lấy tên tất cả intermediate tensors"""
        names = []
        for node in self.model.graph.node:
            for output in node.output:
                names.append(output)
        return names
    
    def collect_activation_stats(self, test_loader, max_samples=50):
        """
        Collect statistics từ activations
        """
        print(f"Collecting activation stats from {max_samples} samples...")
        
        stats = defaultdict(lambda: {
            'min': [], 'max': [], 'mean': [], 'std': [],
            'range': [], 'outlier_ratio': []
        })
        
        # Modify session để output intermediate values
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        # Create session with all outputs
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        modified_model = self._add_intermediate_outputs()
        session = ort.InferenceSession(
            modified_model.SerializeToString(),
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        for i, batch in enumerate(test_loader):
            if i >= max_samples:
                break
            
            mel_feats, mel_lens = batch
            
            # Run inference
            outputs = session.run(None, {
                'mel_spectrogram': mel_feats.numpy(),
                'mel_length': mel_lens.numpy()
            })
            
            # Collect stats
            for j, name in enumerate(self.intermediate_names):
                if j >= len(outputs):
                    break
                
                activation = outputs[j]
                
                min_val = float(np.min(activation))
                max_val = float(np.max(activation))
                mean_val = float(np.mean(activation))
                std_val = float(np.std(activation))
                range_val = float(max_val - min_val)

                stats[name]['min'].append(min_val)
                stats[name]['max'].append(max_val)
                stats[name]['mean'].append(mean_val)
                stats[name]['std'].append(std_val)
                stats[name]['range'].append(range_val)

                # Outlier ratio (values > 3 std from mean)
                outliers = np.abs(activation - mean_val) > 3 * std_val
                stats[name]['outlier_ratio'].append(float(np.mean(outliers)))

        
        # Aggregate stats
        aggregated = {}
        for name, values in stats.items():
            aggregated[name] = {
                'avg_range': float(np.mean(values['range'])),
                'max_range': float(np.max(values['range'])),
                'avg_std': float(np.mean(values['std'])),
                'avg_outlier_ratio': float(np.mean(values['outlier_ratio'])),
            }
        
        return aggregated
    
    def _add_intermediate_outputs(self):
        """Modify ONNX model để output tất cả intermediate tensors"""
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(self.model)
        
        # Add all intermediate tensors as outputs
        for name in self.intermediate_names[:100]:  # Limit để không quá nhiều
            value_info = onnx.ValueInfoProto()
            value_info.name = name
            model_copy.graph.output.append(value_info)
        
        return model_copy
    
    def identify_hard_to_quantize_nodes(self, stats, top_k=20):
        """
        Xác định nodes khó quantize dựa trên stats
        
        Scoring:
        - High range → high score
        - High outlier ratio → high score
        - High std → high score
        """
        scores = {}
        
        # Normalize metrics
        ranges = [s['avg_range'] for s in stats.values()]
        stds = [s['avg_std'] for s in stats.values()]
        outliers = [s['avg_outlier_ratio'] for s in stats.values()]
        
        max_range = max(ranges) if ranges else 1
        max_std = max(stds) if stds else 1
        max_outlier = max(outliers) if outliers else 1
        
        for name, s in stats.items():
            score = (
                0.4 * (s['avg_range'] / max_range) +
                0.3 * (s['avg_std'] / max_std) +
                0.3 * (s['avg_outlier_ratio'] / max_outlier)
            )
            scores[name] = score
        
        # Sort by score
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n=== TOP {top_k} HARD-TO-QUANTIZE NODES ===")
        for i, (name, score) in enumerate(sorted_nodes[:top_k]):
            print(f"{i+1}. {name}")
            print(f"   Score: {score:.3f}")
            print(f"   Range: {stats[name]['avg_range']:.2f}")
            print(f"   Std: {stats[name]['avg_std']:.2f}")
            print(f"   Outliers: {stats[name]['avg_outlier_ratio']:.3f}")
        
        return sorted_nodes[:top_k]
    
    def plot_activation_distributions(self, stats, output_dir='activation_plots'):
        """Vẽ biểu đồ distribution"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot top 10 nodes with highest range
        top_nodes = sorted(
            stats.items(), 
            key=lambda x: x[1]['avg_range'], 
            reverse=True
        )[:10]
        
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        for i, (name, s) in enumerate(top_nodes):
            ax = axes[i]
            
            metrics = ['avg_range', 'avg_std', 'avg_outlier_ratio']
            values = [s[m] for m in metrics]
            
            ax.bar(metrics, values)
            ax.set_title(name.split('/')[-1][:30], fontsize=8)
            ax.tick_params(labelsize=6)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_hard_nodes.png', dpi=150)
        print(f"Saved plot to {output_dir}/top_hard_nodes.png")
    
    def generate_skip_recommendations(self, stats, sensitivity_results=None):
        """
        Kết hợp activation stats với sensitivity results để đề xuất
        """
        print("\n=== FINAL RECOMMENDATIONS ===")
        
        # Hard-to-quantize based on activations
        hard_nodes = self.identify_hard_to_quantize_nodes(stats, top_k=30)
        hard_node_names = [name for name, score in hard_nodes]
        
        # Map tensor names to node names
        node_map = self._map_tensors_to_nodes()
        
        recommended_skip = set()
        for tensor_name in hard_node_names:
            if tensor_name in node_map:
                recommended_skip.add(node_map[tensor_name])
        
        print(f"\nRecommended nodes to skip: {len(recommended_skip)}")
        
        # If có sensitivity results, combine
        if sensitivity_results:
            sensitive_layers = [
                layer for layer, data in sensitivity_results['layer_sensitivity'].items()
                if data['wer_increase'] > 3.0  # WER tăng >3%
            ]
            
            print(f"Sensitive layers (WER increase > 3%): {len(sensitive_layers)}")
            print(f"Combined recommendation: Skip both groups")
        
        return list(recommended_skip)
    
    def _map_tensors_to_nodes(self):
        """Map tensor names to node names"""
        mapping = {}
        for node in self.model.graph.node:
            for output in node.output:
                mapping[output] = node.name
        return mapping


# (Giữ nguyên các hàm class ActivationAnalyzer ở trên)

def main():



    # === CẤU HÌNH CHUẨN ===
    FLOAT_MODEL = "conformer_vie.onnx"
    CONFIG_PATH = "config/phase2.yaml"
    DATASET_NAME = "linhtran92/viet_bud500"
    
    # Load config
    logger.info(f"Loading config from {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)

    # Khởi tạo Preprocessor CHUẨN
    logger.info("Initializing standard preprocessor...")
    preproc_cfg = config.get("preprocessor", {})
    preprocessor = AudioToMelSpectrogramPreprocessor(**preproc_cfg)
    preprocessor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    preprocessor.eval()

    # Chuẩn bị test loader (dùng raw dataset)
    test_ds = load_dataset(DATASET_NAME, split="test")
    test_ds = test_ds.shuffle(seed=42).select(range(50)) # Chỉ 50 samples
    
    @torch.no_grad()
    def data_generator():
        """Generator dùng preprocessor CHUẨN"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_sr = config["dataset"].get("target_sampling_rate", 16000)
        
        for example in test_ds:
            audio_array = example['audio']['array']
            sr = example['audio']['sampling_rate']
            
            waveform = torch.from_numpy(audio_array).float().unsqueeze(0).to(device)
            if sr != target_sr:

                waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
            
            length = torch.tensor([waveform.shape[1]], dtype=torch.long).to(device)
            
            # Chạy preprocessor chuẩn
            mel_feats, mel_lens = preprocessor(waveform, length)
            
            # Chuyển về CPU/numpy cho ONNX runtime
            yield (mel_feats.cpu(), mel_lens.cpu())
    
    # Analyze
    analyzer = ActivationAnalyzer(FLOAT_MODEL)
    
    print("Collecting activation statistics (với preprocessor chuẩn)...")
    stats = analyzer.collect_activation_stats(data_generator(), max_samples=50)
    
    # Identify hard nodes
    hard_nodes = analyzer.identify_hard_to_quantize_nodes(stats)
    
    # Plot
    analyzer.plot_activation_distributions(stats)
    
    # Generate recommendations
    recommendations = analyzer.generate_skip_recommendations(stats)
    
    # Save
    with open('activation_based_skip_list.json', 'w') as f:
        json.dump({
            'recommended_nodes': recommendations,
            'stats': stats
        }, f, indent=2, default=float)
    
    print("\n✅ Analysis complete (đã sửa)! Check 'activation_based_skip_list.json' mới.")


if __name__ == "__main__":
    main()