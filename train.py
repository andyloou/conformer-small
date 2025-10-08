import argparse
from vietasr.asr_task import ASRTask
from loguru import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ASR Model on VIVOS or BUD500 dataset')
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help='Path to config YAML file')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory for checkpoints (override config)')
    parser.add_argument('-d', '--device', type=str, default="cuda", 
                        help='Device: cpu, cuda, cuda:0, cuda:1, etc.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path (override config)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--resume_mode', type=str, default = "selective", choices = ["selective", "full"],
             help='Resume mode: selective (encoder only) or full (encoder + decoder)')

    args = parser.parse_args()
    
    # Log training information
    logger.info("="*60)
    logger.info("ASR Training Script")
    logger.info("="*60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir if args.output_dir else 'from config'}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Resume checkpoint: {args.resume if args.resume else 'None'}")
    logger.info(f"W&B logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    logger.info("="*60)
    
    # Create task
    task = ASRTask(
        config=args.config,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Disable wandb if requested
    if args.no_wandb:
        task.config["train"]["wandb_config"] = None
        logger.info("Weights & Biases logging disabled")
    
    # Override pretrained_path if resume is specified
    if args.resume:
        task.config["train"]["pretrained_path"] = args.resume
        logger.info(f"Will resume training from: {args.resume}")
    
    # Start training
    try:
        task.run_train()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C)")
        logger.info("Saving current state...")
        task.stop_wandb()
        logger.info("Training stopped safely")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        task.stop_wandb()
        raise e
    
#python train.py -c config/conformer.yaml -d cuda 
#python train.py -c config/conformer.yaml -d cuda --resume model_path.pth 
"""
Quy trình train Việt:
Epoch 1-2: Chạy với pretrained_path (English) → Train encoder (fine-tune) + decoder từ đầu.
Dừng (Ctrl+C): Code catch KeyboardInterrupt, save checkpoint.pt (full state tại epoch 2).
Tiếp tục: python train.py -c config/conformer.yaml --resume /path/to/checkpoint.pt -d cuda → Resume từ epoch 3, load full (model đã train Việt, optimizer state để lr đúng vị trí).
Lưu ý: Nếu muốn full load cho resume (bỏ selective), comment if key.startswith('encoder.'): trong model.py's load_checkpoint. Hoặc thêm flag ở config resume_mode: full.
"""
