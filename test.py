import argparse
from vietasr.asr_task import ASRTask
from loguru import logger
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio files using trained ASR model')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input audio file or directory')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output text file (if not specified, print to console)')
    parser.add_argument('-d', '--device', type=str, default="cuda",
                        help='Device: cpu, cuda, cuda:0, cuda:1, etc.')
    
    # Decoding options
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size for decoding')
    parser.add_argument('--kenlm-path', type=str, default=None,
                        help='Path to KenLM language model')
    parser.add_argument('--kenlm-alpha', type=float, default=0.5,
                        help='LM weight (alpha)')
    parser.add_argument('--kenlm-beta', type=float, default=1.5,
                        help='Word insertion bonus (beta)')
    
    args = parser.parse_args()
    
    # Create task
    logger.info(f"Loading model from: {args.model}")
    task = ASRTask(
        config=args.config,
        output_dir=None,
        device=args.device
    )
    
    # Load model
    task.load_checkpoint(args.model)
    
    # Setup beam search decoder if needed
    if args.beam_size > 1 or args.kenlm_path:
        logger.info("Setting up beam search decoder...")
        task.setup_beamsearch(
            kenlm_path=args.kenlm_path,
            kenlm_alpha=args.kenlm_alpha,
            kenlm_beta=args.kenlm_beta,
            beam_size=args.beam_size
        )
    
    # Get audio files
    if os.path.isdir(args.input):
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(glob.glob(os.path.join(args.input, ext)))
        audio_files.sort()
        logger.info(f"Found {len(audio_files)} audio files in directory")
    else:
        audio_files = [args.input]
    
    if len(audio_files) == 0:
        logger.error("No audio files found!")
        exit(1)
    
    # Transcribe
    results = []
    for audio_file in audio_files:
        logger.info(f"Transcribing: {audio_file}")
        try:
            text = task.transcribe(audio_file)
            results.append({
                'file': audio_file,
                'transcription': text
            })
            logger.success(f"  -> {text}")
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_file}: {e}")
            results.append({
                'file': audio_file,
                'transcription': f"ERROR: {e}"
            })
    
    # Save or print results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{result['file']}|{result['transcription']}\n")
        logger.success(f"Results saved to: {args.output}")
    else:
        logger.info("\n" + "="*60)
        logger.info("Transcription Results:")
        logger.info("="*60)
        for result in results:
            print(f"{os.path.basename(result['file'])}: {result['transcription']}")
