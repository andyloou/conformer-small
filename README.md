# VietASR: Vietnamese Speech Recognition Training Framework

This repository implements a robust training and inference framework for Vietnamese Automatic Speech Recognition (ASR). It is designed to train models like **Conformer-CTC** using modern techniques such as Automatic Mixed Precision (AMP), Differential Learning Rates, and Spectrogram Augmentation.

## 🌟 Features

* **Model Support**: Conformer-CTC (and extensible for QuartzNet/Jasper).
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

## 📂 Project Structure
## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/VietASR.git](https://github.com/yourusername/VietASR.git)
    cd VietASR
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)  # Adjust for your CUDA version
    pip install -r requirements.txt
    ```
    *Required libraries include: `loguru`, `wandb`, `librosa`, `pyctcdecode`, `numpy`, etc.*

## 🚀 Training

### 1. Configuration
Modify `config/conformer.yaml` to set your parameters. Key settings:
* `freeze_encoder`: Set `True` for the first phase (train decoder only).
* `use_huggingface`: Set `True` to use online datasets like VIVOS or BUD500.

### 2. Basic Training Command
```bash
python train.py -c config/conformer.yaml -d cuda
