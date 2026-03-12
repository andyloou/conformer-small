# apply_sensitivity.py

import json
import os
import onnx
import vai_q_onnx
import torch
from loguru import logger

# Import các component chuẩn
from utils import load_config
from quan import MelSpecDataReader

def _extract_all_nodes_and_outputs(model_path):
    """
    Trích xuất TẤT CẢ nodes và output tensors của chúng
    (Copy từ sensitivity_analysis.py)
    """
    model = onnx.load(model_path)
    nodes_and_tensors_map = {}
    for node in model.graph.node:
        names_to_skip = [node.name] + list(node.output)
        nodes_and_tensors_map[node.name] = names_to_skip
    return nodes_and_tensors_map

def _group_nodes_by_layer(all_nodes_map):
    """
    Nhóm TẤT CẢ nodes VÀ TENSORS theo layer
    (Copy từ sensitivity_analysis.py)
    """
    layer_groups = {}
    for node_name, names_to_skip_list in all_nodes_map.items():
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
    
    final_groups = {idx: list(names) for idx, names in layer_groups.items()}
    return final_groups

def create_optimized_quantized_model(
    json_path, 
    float_model_path, 
    config_path, 
    output_path, 
    top_n=3
):
    """
    Tạo model quantize cuối cùng bằng cách skip N layers nhạy cảm nhất.
    """
    
    # 1. Load file kết quả sensitivity
    logger.info(f"Loading sensitivity results from {json_path}")
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # 2. Tìm Top N layer nhạy cảm nhất (có wer_increase THẤP NHẤT)
    sensitivities = []
    for layer_name, data in results['layer_sensitivity'].items():
        if 'wer_increase' in data:
            sensitivities.append((layer_name, data['wer_increase']))
    
    # Sắp xếp theo wer_increase TĂNG DẦN
    sensitivities.sort(key=lambda x: x[1], reverse=False)
    
    top_n_sensitive_layers = [name for name, wer in sensitivities[:top_n]]
    
    # === THÊM MỚI: Kết hợp cả hai phương pháp ===
    top_n_sensitive_layers.append('pre_encode')
    top_n_sensitive_layers = list(set(top_n_sensitive_layers)) # Đảm bảo không trùng lặp
    # ==========================================

    logger.success(f"Combined sensitive layers to skip: {top_n_sensitive_layers}")
    
    # 3. Tái tạo lại các nhóm node từ model float
    # (Vì file JSON không lưu danh sách node đầy đủ)
    logger.info(f"Regenerating all node groups from {float_model_path}...")
    all_nodes_map = _extract_all_nodes_and_outputs(float_model_path)
    all_layer_groups = _group_nodes_by_layer(all_nodes_map)
    
    # 4. Tạo danh sách exclude cuối cùng
    final_exclude_list = set()
    for layer_name in top_n_sensitive_layers:
        if layer_name in all_layer_groups:
            num_nodes = len(all_layer_groups[layer_name])
            logger.info(f"Adding {num_nodes} nodes/tensors from '{layer_name}' to exclusion list.")
            final_exclude_list.update(all_layer_groups[layer_name])
        else:
            logger.warning(f"Layer '{layer_name}' from JSON not found in regenerated groups. Skipping.")
    
    logger.success(f"Total unique nodes/tensors to exclude: {len(final_exclude_list)}")
    
    if not final_exclude_list:
        logger.error("Exclusion list is empty. Aborting.")
        return

    # 5. Load config
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # 6. Khởi tạo DataReader
    logger.info("Initializing calibration data reader...")
    calibration_data_reader = MelSpecDataReader(
        float_model_path,
        config=config,
        dataset_name="linhtran92/viet_bud500",
        max_samples=100, # 100-200 samples là đủ cho calibration
        use_huggingface=True,
        split="test",
        max_duration=10.0
    )
    
    extra_options = {
        'ActivationSymmetric': True,
        'WeightSymmetric': True,
        'AddQDQPairToWeight': True
    }
    
    logger.info(f"Starting final quantization... output: {output_path}")
    try:
        vai_q_onnx.quantize_static(
            model_input=float_model_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=vai_q_onnx.QuantFormat.QDQ, # Dùng QDQ
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            nodes_to_exclude=list(final_exclude_list), # Chuyển set thành list
            extra_options=extra_options
        )
        logger.success(f"Successfully created optimized quantized model at {output_path}")
    except Exception as e:
        logger.error(f"Final quantization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # === CẤU HÌNH ===
    JSON_PATH = "layer_sensitivity.json"
    FLOAT_MODEL = "conformer_vie_vivos_vlsp.onnx"
    CONFIG_PATH = "config/phase2.yaml"
    FINAL_OUTPUT_PATH = "conformer_quantized_vivos_vlsp.onnx"
    TOP_N_LAYERS_TO_SKIP = 3
    
    # ================
    
    create_optimized_quantized_model(
        json_path=JSON_PATH,
        float_model_path=FLOAT_MODEL,
        config_path=CONFIG_PATH,
        output_path=FINAL_OUTPUT_PATH,
        top_n=TOP_N_LAYERS_TO_SKIP
    )