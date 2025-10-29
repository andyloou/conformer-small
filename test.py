# evaluate.py (Đã sửa lỗi)
import argparse
from vietasr.asr_task import ASRTask
from loguru import logger
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ASR model on a test set')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('-d', '--device', type=str, default="cuda",
                        help='Device: cpu, cuda, cuda:0, etc.')
    
    # Options for dataset source
    parser.add_argument('--use-hf', action='store_true',
                        help='Use a dataset from Hugging Face Hub.')
    parser.add_argument('--dataset-name', type=str, default="linhtran92/viet_bud500",
                        help='Name of the Hugging Face dataset (e.g., "linhtran92/viet_bud500").')
    parser.add_argument('--test-meta-file', type=str, default=None,
                        help='Path to the meta file for a local test set.')
    
    # SỬA Ở ĐÂY: Đặt tất cả default thành None để ưu tiên đọc config
    parser.add_argument('--beam-size', type=int, default=None,
                        help='Beam size. Overrides config if set.')
    parser.add_argument('--kenlm-path', type=str, default=None,
                        help='KenLM path. Overrides config if set.')
    parser.add_argument('--kenlm-alpha', type=float, default=None,
                        help='LM weight (alpha). Overrides config if set.')
    parser.add_argument('--kenlm-beta', type=float, default=None,
                        help='Word insertion bonus (beta). Overrides config if set.')
    
    args = parser.parse_args()
    
    # Create task
    logger.info(f"Loading model from: {args.model}")
    task = ASRTask(
        config=args.config,
        output_dir=None,
        device=args.device
    )
    
    # Load model checkpoint
    task.load_checkpoint(args.model)
    
    # SỬA Ở ĐÂY: Luôn gọi setup_beamsearch để nó tự đọc config
    # Nó sẽ tự động lấy giá trị từ config nếu các tham số dòng lệnh không được cung cấp.
    logger.info("Setting up beam search decoder (reading from config if args not set)...")
    task.setup_beamsearch(
        kenlm_path=args.kenlm_path,
        kenlm_alpha=args.kenlm_alpha,
        kenlm_beta=args.kenlm_beta,
        beam_size=args.beam_size
    )
    
    # Xác định nguồn dữ liệu test (ưu tiên dòng lệnh -> config)
    use_hf = args.use_hf
    dataset_name = args.dataset_name
    test_meta_filepath = args.test_meta_file

    if not use_hf and not test_meta_filepath:
        if task.config["dataset"].get("use_huggingface", False):
            use_hf = True
            dataset_name = task.config["dataset"].get("dataset_name")
            logger.info(f"Using HuggingFace dataset from config: {dataset_name}")
        elif task.config["dataset"].get("test_filepath"):
            test_meta_filepath = task.config["dataset"].get("test_filepath")
            logger.info(f"Using local test meta file from config: {test_meta_filepath}")
        else:
            logger.error("No test data source specified in command-line args or config file!")
            exit(1)

    # Run the evaluation
    task.run_test(
        test_meta_filepath=test_meta_filepath,
        use_huggingface=use_hf,
        dataset_name=dataset_name
    )
