# VietASR: Vietnamese Speech Recognition Training Framework

This repository implements a robust training and inference framework for Vietnamese Automatic Speech Recognition (ASR). It is designed to train models like **Conformer-CTC** using modern techniques such as Automatic Mixed Precision (AMP), Differential Learning Rates, and Spectrogram Augmentation.

## 🌟 Features

* **Model Support**: Conformer-CTC-small.
* **Training Strategies**:
    * **AMP (Automatic Mixed Precision)**: Faster training with `torch.cuda.amp`.
    * **Smart Resume**: Supports "Selective" (load weights only) and "Full" (load weights + optimizer states) resume modes.
    * **Freeze/Unfreeze**: Two-phase training strategy (Freeze Encoder first, then Full Fine-tuning).
    * **Differential Learning Rates**: different LRs for Encoder and Decoder.
* **Decoding**:
    * Standard Greedy Search.
    * **Beam Search** with KenLM language model support (using `pyctcdecode`).
* **Logging**: Integrated with **Weights & Biases (WandB)** and Tensorboard.
* **Data**: Supports HuggingFace datasets (e.g., `linhtran92/viet_bud500`) or local metadata files.


## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/VietASR.git](https://github.com/yourusername/VietASR.git)
    cd VietASR
    ```

2.  **Install dependencies:**
    ```bash
     #pytorch == 2.6.0
      pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
     #onnxruntime ==1.5.1
      pip install onnxruntime==1.5.1 onnx==1.14.1    
    ```
    *Required libraries include: `loguru`, `wandb`, `librosa`, `pyctcdecode`, `numpy`.
    If you want to use Vitis AI quantization:
    ```bash
    git clone https://github.com/Xilinx/Vitis-AI.git
    cd Vitis-AI/src/vai_quantizer/vai_q_onnx
    sh build.sh
    pip install pkgs/*.whl
    ```
    

## 🚀 Training

### 1. Configuration
Modify `config/conformer.yaml` to set your parameters. Key settings:
* `freeze_encoder`: Set `True` for the first phase (train decoder only).
* `use_huggingface`: Set `True` to use online datasets like VIVOS or BUD500.

### 2. Basic Training Command
```bash
python train.py -c config/conformer.yaml -d cuda

```
### 2. Testing model after training
```bash
python test.py -c config/conformer.yaml -d cuda
```

