import editdistance
import yaml
from loguru import logger

def calculate_wer(preds: str, targets: str, use_cer=False) -> float:
    """Calculate sentence-level WER score.

    Args:
        pred: Prediction character sequences. (B, ?)
        target: Target character sequences. (B, ?)
        use_cer: bool

    Returns:
        : Average WER score

    """

    distances, lens = [], []

    for pred, target in zip(preds, targets):
        if use_cer:
            pred = list(pred)
            target = list(target)
        else:
            pred = pred.split()
            target = target.split()

        distances.append(editdistance.eval(pred, target))
        lens.append(len(target))

    return float(sum(distances)) / sum(lens) * 100

def load_config(config_path: str) -> dict:
    """Tải file config YAML."""
    logger.info(f"Loading config from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.success("Config loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise
