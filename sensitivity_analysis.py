import onnx
import onnxruntime as ort
import numpy as np
import torch
import torchaudio
from datasets import load_from_disk
import json
from tqdm import tqdm
from vai_q_onnx import quantize_static, PowerOfTwoMethod
import os
from loguru import logger
from torch.utils.data import DataLoader
import vai_q_onnx

try:
    from utils import load_config
    from vietasr.model import AudioToMelSpectrogramPreprocessor
    from vietasr.dataset.dataset import ASRDataset, ASRCollator
    from vietasr.utils.utils import calculate_wer
except ImportError:
    logger.error("Cannot import 'vietasr' or 'utils'. Please run in the correct environment.")
    exit(1)


class LayerSensitivityAnalyzer:
    def __init__(self, float_model_path, config: dict, test_dataset):
        self.float_model_path = float_model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        preproc_cfg = self.config.get("preprocessor", {})
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preproc_cfg)
        self.preprocessor.to(self.device)
        self.preprocessor.eval()

        self.collator = ASRCollator(
            bpe_model_path=self.config["dataset"]["bpe_model_path"],
            target_sampling_rate=self.config["dataset"].get("target_sampling_rate", 16000)
        )
        self.vocab = self.collator.get_vocab()
        self.blank_id = len(self.vocab)

        test_asr_dataset = ASRDataset(hf_dataset=test_dataset)
        self.dataloader = DataLoader(
            dataset=test_asr_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=self.collator
        )

        self.float_session = ort.InferenceSession(float_model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        self.all_nodes_map = self._extract_all_nodes_and_outputs()

    def _extract_all_nodes_and_outputs(self):
        model = onnx.load(self.float_model_path)
        nodes_and_tensors_map = {}
        for node in model.graph.node:
            nodes_and_tensors_map[node.name] = [node.name] + list(node.output)
        return nodes_and_tensors_map

    def _group_nodes_by_layer(self):
        layer_groups = {}
        for node_name, names_to_skip_list in self.all_nodes_map.items():
            layer_idx = None
            if '/encoder/layers.' in node_name:
                for part in node_name.split('/'):
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

        return {idx: list(names) for idx, names in layer_groups.items()}

    def quantize_except_nodes(self, nodes_to_skip, output_path):
        from quan import MelSpecDataReader

        calibration_data_reader = MelSpecDataReader(
            self.float_model_path,
            config=self.config,
            dataset_name="/home/datasets/viet_bud500",
            max_samples=100,
            use_huggingface=True,
            split="test",
            max_duration=10.0
        )

        quantize_static(
            model_input=self.float_model_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            calibrate_method=PowerOfTwoMethod.NonOverflow,
            nodes_to_exclude=nodes_to_skip,
            extra_options={
                'ActivationSymmetric': True,
                'WeightSymmetric': True,
                'AddQDQPairToWeight': True
            }
        )

    def evaluate_model(self, model_path):
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_path}: {e}")
            return 100.0

        all_predictions = []
        all_labels = []

        for batch in tqdm(self.dataloader, desc=f"Evaluating {os.path.basename(model_path)}"):
            try:
                audio = batch[0].to(self.device)
                audio_lens = batch[1].to(self.device)

                with torch.no_grad():
                    mel_feats, mel_lens = self.preprocessor(audio, audio_lens)

                log_probs_onnx, encoded_length_onnx = session.run(None, {
                    'mel_spectrogram': mel_feats.cpu().numpy(),
                    'mel_length': mel_lens.cpu().numpy()
                })

                pred_ids = self._greedy_decode(log_probs_onnx[0, :encoded_length_onnx[0]])
                all_predictions.append(self.collator.ids2text(pred_ids))

                targets, target_lens = batch[2], batch[3]
                all_labels.append(self.collator.ids2text(targets[0, :target_lens[0]].tolist()))

            except Exception as e:
                logger.warning(f"Error processing batch: {e}. Skipping.")
                continue

        if not all_labels:
            logger.error("Evaluation failed: no samples processed.")
            return 100.0

        return calculate_wer(all_predictions, all_labels, use_cer=False)

    def _greedy_decode(self, log_probs):
        result = []
        prev = self.blank_id
        for t in np.argmax(log_probs, axis=1):
            if t != self.blank_id and t != prev:
                result.append(int(t))
            prev = t
        return result

    def analyze_layer_sensitivity(self, output_json='layer_sensitivity.json'):
        baseline_wer = self.evaluate_model(self.float_model_path)

        if baseline_wer >= 99.0:
            logger.error(f"Baseline WER is {baseline_wer:.2f}%. Check your dataset/model setup.")
            return {}

        layer_groups = self._group_nodes_by_layer()
        results = {'baseline_wer': baseline_wer, 'layer_sensitivity': {}}

        for layer_name, nodes_to_skip in sorted(layer_groups.items()):
            temp_model = f'temp_skip_{layer_name}.onnx'
            try:
                self.quantize_except_nodes(nodes_to_skip, temp_model)
                layer_wer = self.evaluate_model(temp_model)
                wer_increase = layer_wer - baseline_wer
                results['layer_sensitivity'][layer_name] = {
                    'wer': layer_wer,
                    'wer_increase': wer_increase,
                    'num_nodes': len(nodes_to_skip),
                }
            except Exception as e:
                logger.error(f"Error processing {layer_name}: {e}")
                results['layer_sensitivity'][layer_name] = {'error': str(e)}
            finally:
                if os.path.exists(temp_model):
                    os.remove(temp_model)

        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

        self._recommend_skip_layers(results)
        logger.info(f"Analysis complete. Results saved to {output_json}")

        return results

    def _recommend_skip_layers(self, results):
        sensitivities = [
            (layer, data['wer_increase'])
            for layer, data in results['layer_sensitivity'].items()
            if 'wer_increase' in data
        ]

        if not sensitivities:
            logger.warning("No valid sensitivity data to analyze.")
            return

        sensitivities.sort(key=lambda x: x[1])

        logger.info(f"Baseline WER: {results['baseline_wer']:.2f}%")
        logger.info("Most sensitive layers (recommend skipping):")
        for i, (layer, wer_inc) in enumerate(sensitivities[:5]):
            logger.info(f"  {i+1}. {layer}: +{wer_inc:.2f}%")

        logger.info("Least sensitive layers (safe to quantize):")
        for i, (layer, wer_inc) in enumerate(reversed(sensitivities[-5:])):
            logger.info(f"  {i+1}. {layer}: +{wer_inc:.2f}%")


def main():
    FLOAT_MODEL = "conformer_vie.onnx"
    DATASET_PATH = "/home/datasets/viet_bud500"
    CONFIG_PATH = "config/phase2.yaml"

    config = load_config(CONFIG_PATH)

    full_ds = load_from_disk(DATASET_PATH)
    test_ds = full_ds["test"].select(range(50))

    analyzer = LayerSensitivityAnalyzer(FLOAT_MODEL, config, test_ds)
    analyzer.analyze_layer_sensitivity()


if __name__ == "__main__":
    main()