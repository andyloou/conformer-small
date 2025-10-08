import os
from typing import Union
from loguru import logger
import torch
import wandb
from torch.optim import Adam, AdamW
from torch.utils.data.dataloader import DataLoader
from loguru import logger
import numpy as np
from utils import load_config, save_config
from vietasr.dataset.dataset import ASRDataset, ASRCollator
from vietasr.model import ConformerCTC as ASRModel, AudioToMelSpectrogramPreprocessor
from vietasr.utils.lr_scheduler import WarmupLR
from vietasr.utils.utils import calculate_wer
import torch.cuda.amp as amp

class ASRTask():
    def __init__(self, config: str, output_dir: str=None, device: str="cpu", resume_mode: str= "selective") -> None:

        config = load_config(config)
        self.resume_mode = config["train"].get("resume_mode", resume_mode)
        self.collate_fn = ASRCollator(
            bpe_model_path=config["dataset"]["bpe_model_path"],
            target_sampling_rate=config["dataset"].get("target_sampling_rate", 16000)  # Thêm tham số này
        )
        self.vocab = self.collate_fn.get_vocab()
        model = ASRModel(vocab_size=len(self.vocab), pad_id=self.collate_fn.pad_id, **config["model"])

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = config["train"]["output_dir"]

        self.device = torch.device(device)
        preproc_cfg = config.get("preprocessor", {})
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**preproc_cfg)

        self.config = config
        self.use_amp = self.config["train"].get("use_amp", False)
        self.model = model
        self.ctc_decoder = None
        self.optimizer = None
        self.lr_scheduler = None
        self.epoch = None
        
    def init_wandb(self):
        try:
            wandb.login()
        except wandb.errors.UsageError:
            logger.info("wandb not configured! run `wandb login` to enable")

        if self.config["train"].get("wandb_config"):
            os.makedirs(f"{self.output_dir}/tensorboard", exist_ok=True)
            wandb.tensorboard.patch(root_logdir=f"{self.output_dir}/tensorboard")
            wandb.init(
                config=self.config,
                project=self.config["train"]["wandb_config"]["project"],
                name=self.config["train"]["wandb_config"]["tag"],
                id=self.config["train"]["wandb_config"]["tag"],
                dir=self.config["train"]["output_dir"],
                resume="allow",
                sync_tensorboard=True
            )
            
    def stop_wandb(self):
        if wandb.run:
            wandb.finish()

    def train_one_epoch(self) -> dict:
        self.model.train()  # SpecAug sẽ apply ở model forward (nếu training)

        train_loss_epoch = 0
        ctc_loss_epoch = 0
        decoder_loss_epoch = 0

        dataloader = DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_worker,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )

        #num_batch = len(dataloader)
        batch_count = 0
        self.optimizer.zero_grad()

        # THÊM: AMP scaler với device explicit
        scaler = torch.cuda.amp.GradScaler(init_scale=2.**16, enabled=self.use_amp)

        for i, batch in enumerate(dataloader):
            # Giữ nguyên preprocess: audio to mel
            batch_count += 1
            audio = batch[0].to(self.device)
            audio_lens = batch[1].to(self.device)
            mel_feats, mel_lens = self.preprocessor(audio, audio_lens)
            
            # Targets giữ nguyên
            targets = batch[2].to(self.device)
            target_lens = batch[3].to(self.device)

            # Forward với AMP autocast
            if self.use_amp:
                with torch.cuda.amp.autocast(enabled = True):  # Explicit device cho autocast
                    retval = self.model(mel_feats, mel_lens, targets, target_lens)
                    loss = retval["loss"]
            else:
                retval = self.model(mel_feats, mel_lens, targets, target_lens)
                loss = retval["loss"]
            
            if torch.isnan(loss):
                self.optimizer.zero_grad()
                continue
            
            loss = loss / self.acc_steps
            
            # Backward với AMP scaler
            if self.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)  # Unscale trước clip
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                scaler.step(self.optimizer)
                scaler.update()
                if (i + 1) % self.acc_steps == 0:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                if (i + 1) % self.acc_steps == 0:
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            train_loss = loss.detach().item()
            train_loss_epoch += train_loss
            ctc_loss = retval["ctc_loss"].detach().item()
            ctc_loss_epoch += ctc_loss
            decoder_loss = retval["decoder_loss"].detach().item()
            decoder_loss_epoch += decoder_loss
            
            if (i + 1) % 100 == 0:
                logger.info(f"[TRAIN] EPOCH {self.epoch} | BATCH {i+1} | loss={train_loss} | ctc_loss={ctc_loss} | decoder_loss={decoder_loss}")
                predicts = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
                labels = self.model.get_labels(targets, target_lens)
                logger.warning(f"+ Label  : {self.collate_fn.ids2text(labels[0].tolist())}")
                logger.warning(f"+ Predict: {self.collate_fn.ids2text(predicts[0].tolist())}")
                wandb.log(
                    {
                        "train": {
                            "train_loss": train_loss,
                            "ctc_loss": ctc_loss,
                            "decoder_loss": decoder_loss
                        },
                        "step": (self.epoch - 1)  + i + 1
                    }
                )

        train_stats = {
            "train_loss": train_loss_epoch / batch_count,
            "train_ctc_loss": ctc_loss_epoch / batch_count,
            "train_decoder_loss": decoder_loss_epoch / batch_count,
        } 
        return train_stats

    def valid_one_epoch(self) -> dict:
        valid_loss_epoch = 0
        ctc_loss_epoch = 0
        decoder_loss_epoch = 0
        predicts = []
        labels = []

        dataloader = DataLoader(
            dataset=self.valid_dataset,
            num_workers=self.num_worker,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        #num_batch = len(dataloader)
        batch_count = 0
        self.model.eval()  # No spec_aug

        for i, batch in enumerate(dataloader):
            # Giữ nguyên preprocess
            batch_count += 1
            audio = batch[0].to(self.device)
            audio_lens = batch[1].to(self.device)
            mel_feats, mel_lens = self.preprocessor(audio, audio_lens)
            
            targets = batch[2].to(self.device)
            target_lens = batch[3].to(self.device)

            with torch.no_grad():
                # Forward với AMP autocast (optional cho valid, nhưng consistent)
                if self.use_amp:
                    with torch.cuda.amp.autocast(enabled=True):
                        retval = self.model(mel_feats, mel_lens, targets, target_lens)
                        loss = retval["loss"]
                else:
                    retval = self.model(mel_feats, mel_lens, targets, target_lens)
                    loss = retval["loss"]

            valid_loss = loss.detach().item()
            valid_loss_epoch += valid_loss
            ctc_loss = retval["ctc_loss"].detach().item()
            ctc_loss_epoch += ctc_loss
            decoder_loss = retval["decoder_loss"].detach().item()
            decoder_loss_epoch += decoder_loss

            predict = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
            label = self.model.get_labels(targets, target_lens)
            predict_str = [self.collate_fn.ids2text(x) for x in predict]
            label_str = [self.collate_fn.ids2text(x) for x in label]
            predicts += predict_str
            labels += label_str
            
            if (i + 1) % 100 == 0:
                logger.info(f"[VALID] EPOCH {self.epoch} | BATCH {i+1} | loss={valid_loss} | ctc_loss={ctc_loss} | decoder_loss={decoder_loss}")
                logger.warning(f"+ Label  : {label_str[0]}")
                logger.warning(f"+ Predict: {predict_str[0]}")

        valid_stats = {
            "valid_loss": valid_loss_epoch / batch_count,
            "valid_ctc_loss": ctc_loss_epoch / batch_count,
            "valid_decoder_loss": decoder_loss_epoch / batch_count,
            "valid_wer": calculate_wer(predicts, labels),
            "valid_cer": calculate_wer(predicts, labels, use_cer=True)
        }
        return valid_stats

    def load_checkpoint(self, checkpoint_path: str):
        """Load full checkpoint (model + optimizer + etc.) for resume"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract state_dict (handle structure)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        # Gọi model selective load (nhưng cho resume, có thể load full bằng cách comment selective ở model.py)
        optimizer_state, lr_scheduler_state, saved_epoch = self.model.load_checkpoint(checkpoint_path, resume_mode = self.resume_mode)  # Sử dụng model method
        
        # Load optimizer/lr_scheduler nếu có
        if optimizer_state and self.optimizer:
            self.optimizer.load_state_dict(optimizer_state)
        if lr_scheduler_state and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
        if saved_epoch and self.epoch is not None:
            self.epoch = int(saved_epoch)
        
        self.model.to(self.device)
        logger.success(f"Loaded full checkpoint from: {checkpoint_path} with mode: {self.resume_mode}")

    def run_train(self):

        logger.info("="*40)
        logger.info(f"START TRAINING ASR MODEL")
        logger.info("="*40)
        logger.info(f"Config: {self.config}")

        self.num_epoch = self.config["train"]["num_epoch"]
        self.acc_steps = self.config["train"]["acc_steps"]
        self.batch_size = self.config["dataset"]["batch_size"]
        self.num_worker = self.config["dataset"]["num_worker"]
        self.epoch = 0
        self.valid_loss_best = 1000000

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["train"].get("lr", 1e-4),
            weight_decay=self.config["train"].get("weight_decay")
        )
        self.lr_scheduler = WarmupLR(
            self.optimizer,
            warmup_steps=self.config["train"].get("warmup_steps", 25000)
        )

        self.model.to(self.device)
        self.preprocessor.to(self.device)
    
        pretrained_path = self.config["train"].get("pretrained_path")
        if pretrained_path:
            # Cho pretrained: Chỉ load model (không optimizer/epoch, vì initial)
            self.model.load_checkpoint(pretrained_path,resume_mode = self.resume_mode)
            logger.success(f"Loaded pretrained encoder from: {pretrained_path} with mode: {self.resume_mode}")

        # THAY ĐỔI CHÍNH Ở ĐÂY: Load dataset từ HuggingFace thay vì file meta
        dataset_config = self.config["dataset"]
        
        # Kiểm tra xem có dùng HuggingFace dataset hay file meta
        if dataset_config.get("use_huggingface", False):
            # Load từ HuggingFace
            dataset_name = dataset_config.get("dataset_name", "linhtran92/viet_bud500")
            max_duration = dataset_config.get("max_duration", 20.0)
            
            logger.info(f"Loading HuggingFace dataset: {dataset_name}")
            self.train_dataset = ASRDataset(
                dataset_name=dataset_name,
                split="train",
                max_duration=max_duration
            )
            self.valid_dataset = ASRDataset(
                dataset_name=dataset_name,
                split="validation",
                max_duration=max_duration
            )
        else:
            # Load từ file meta (cách cũ)
            logger.info("Loading dataset from meta files")
            self.train_dataset = ASRDataset(meta_filepath=dataset_config["train_filepath"])
            self.valid_dataset = ASRDataset(meta_filepath=dataset_config["valid_filepath"])

        os.makedirs(self.output_dir, exist_ok=True)
        save_config(self.config, os.path.join(self.output_dir, "config.yaml"))
        
        self.init_wandb()
        
        valid_loss_best = self.valid_loss_best
        valid_acc_best = 0

        for epoch in range(self.epoch or 0, self.num_epoch):
            self.epoch = epoch + 1
            logger.info(f"[TRAIN] EPOCH {epoch + 1}/{self.num_epoch} START")
            train_stats = self.train_one_epoch()
            logger.success(f"[TRAIN] STATS: {train_stats}")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "epoch": self.epoch,
                    "valid_loss_best": valid_loss_best
                },
                f"{self.output_dir}/checkpoint.pt")
            torch.save({"model": self.model.state_dict()}, f"{self.output_dir}/epoch_{self.epoch}.pt")
                
            logger.info(f"[TRAIN] EPOCH {epoch + 1}/{self.num_epoch} DONE, Save checkpoint to: {self.output_dir}/checkpoint_epoch_{self.epoch}.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} START")
            valid_stats = self.valid_one_epoch()
            logger.success(f"[VALID] STATS: {valid_stats}")
            valid_loss = valid_stats["valid_loss"]

            if valid_loss < valid_loss_best:
                valid_loss_best = valid_loss
                torch.save({"model": self.model.state_dict()}, f"{self.output_dir}/valid_loss_best.pt")
                logger.success(f"saved best model to {self.output_dir}/valid_loss_best.pt")

            logger.info(f"[VALID] EPOCH {epoch + 1}/{self.num_epoch} DONE")

            wandb.log({"train": train_stats, "valid": valid_stats, "epoch": self.epoch}, commit=True)

        self.stop_wandb()
        logger.success(f"TRAINING ASR MODEL DONE!")

    def run_test(
            self,
            test_meta_filepath: str = None,
            use_huggingface: bool = False,
            dataset_name: str = None,
        ):
        
        logger.info("="*40)
        logger.info(f"START TESTING ASR MODEL")
        logger.info("="*40)
        logger.info(f"+ device: {self.device}")
        logger.info(f"+ Config: {self.config}")
        
        batch_size = self.config["dataset"]["batch_size"]
        num_worker = self.config["dataset"]["num_worker"]
        
        if use_huggingface:
            if dataset_name is None:
                dataset_name = self.config["dataset"].get("dataset_name", "linhtran92/viet_bud500")
            max_duration = self.config["dataset"].get("max_duration", 20.0)
            logger.info(f"Loading test set from HuggingFace: {dataset_name}")
            test_dataset = ASRDataset(
                dataset_name=dataset_name,
                split="test",
                max_duration=max_duration
            )
        else:
            logger.info(f"Loading test set from meta file: {test_meta_filepath}")
            test_dataset = ASRDataset(meta_filepath=test_meta_filepath)
        
        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_worker,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )

        test_loss_total = 0
        test_ctc_loss_total = 0
        test_decoder_loss_total = 0
        test_decoder_acc_total = 0
        predicts = []
        labels = []
        
        #num_batch = len(dataloader)
        batch_count = 0
        self.model.eval()

        for i, batch in enumerate(dataloader):
            batch_count += 1
            audio = batch[0].to(self.device)
            audio_lens = batch[1].to(self.device)
            mel_feats, mel_lens = self.preprocessor(audio, audio_lens)
            
            targets = batch[2].to(self.device)
            target_lens = batch[3].to(self.device)

            with torch.no_grad():
                retval = self.model(mel_feats, mel_lens, targets, target_lens)  # Dùng mel
            loss = retval["loss"]


            test_loss = loss.detach().item()
            test_loss_total += test_loss
            test_ctc_loss = retval["ctc_loss"].detach().item()
            test_ctc_loss_total += test_ctc_loss
            test_decoder_loss = retval["decoder_loss"].detach().item()
            test_decoder_loss_total += test_decoder_loss

            if self.ctc_decoder is not None:
                encoder_out = retval["encoder_out"]
                encoder_out_lens = retval["encoder_out_lens"]
                predict_str = []
                for j in range(encoder_out.shape[0]):
                    predict_str.append(self.ctc_beamsearch(encoder_out[j].unsqueeze(0), encoder_out_lens[j].unsqueeze(0)))
            else:
                predict = self.model.get_predicts(retval["encoder_out"], retval["encoder_out_lens"])
                predict_str = [self.collate_fn.ids2text(x) for x in predict]
            
            predicts += predict_str

            label = self.model.get_labels(batch[2], batch[3])
            label_str = [self.collate_fn.ids2text(x) for x in label]
            labels += label_str
            
            if (i + 1) % 10 == 0:
                logger.info(f"[TEST] BATCH {i+1}| loss={test_loss} | ctc_loss={test_ctc_loss} | decoder_loss={test_decoder_loss}")
                logger.warning(f"+ Label  : {label_str[0]}")
                logger.warning(f"+ Predict: {predict_str[0]}")
        
        wer = calculate_wer(predicts, labels)
        cer = calculate_wer(predicts, labels, use_cer=True)
        
        logger.success(f"Test done.")
        logger.success(f" + CER={cer}%")
        logger.success(f" + WER={wer}%")

    def setup_beamsearch(
        self,
        kenlm_path: str=None,
        word_vocab_path: str=None,
        kenlm_alpha: float=None,
        kenlm_beta: float=None,
        beam_size: int=2,
    ):
        if not kenlm_path:
            kenlm_path = self.config["decode"].get("kenlm_path")
        if not word_vocab_path:
            word_vocab_path = self.config["decode"].get("word_vocab_path")
        if not kenlm_alpha:
            kenlm_alpha = self.config["decode"].get("kenlm_alpha")
        if not kenlm_beta:
            kenlm_beta = self.config["decode"].get("kenlm_beta")
        if not beam_size:
            beam_size = self.config["decode"].get("beam_size")

        if beam_size > 1 and not kenlm_path:
            logger.error(f"must pass --kenlm_path (or set in config file) for language model, if beamsize > 1")
            exit()

        self.beam_size = beam_size

        from pyctcdecode import build_ctcdecoder

        self.ctc_decoder = build_ctcdecoder(
            self.vocab,
            kenlm_model_path=kenlm_path,
            unigrams=word_vocab_path,
            alpha=kenlm_alpha,
            beta=kenlm_beta
        )
        logger.success("Setup ctc decoder done")

    def ctc_beamsearch(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
        )->str:
        
        encoder_out = encoder_out[:, : encoder_out_lens[0], :]
        assert len(encoder_out.shape) == 3, encoder_out.shape
        assert encoder_out.shape[0] == 1, encoder_out.shape
        assert encoder_out.shape[1] == encoder_out_lens[0]

        with torch.no_grad():
            log_probs = self.model.decoder(encoder_out)  # [1, T, C+1] log-softmax đã có
            logit = log_probs.detach().cpu().squeeze(0).numpy()
        text = self.ctc_decoder.decode(
            logits=logit,
            beam_width=self.beam_size
        )
        text = text.replace("<blank>", "").strip()
        return text

    def transcribe(self, _input: Union[str, np.array, torch.Tensor]) -> str:
        if isinstance(_input, str):
            import librosa
            _input = librosa.load(_input, sr=16000, mono=True)[0]  # Raw audio
            _input = torch.from_numpy(_input).float()
        elif isinstance(_input, np.array):
            _input = torch.from_numpy(_input).float()
        elif isinstance(_input, torch.Tensor):
            _input = _input.float()
        else:
            raise NotImplementedError
        if len(_input.shape) == 1:
            _input = _input.unsqueeze(0)  # [1, T_audio]

        length = torch.LongTensor([_input.shape[1]])  # [1]

        _input = _input.to(self.device)
        length = length.to(self.device)

        self.model.eval()

        with torch.no_grad():
            # THÊM MỚI: Preprocess raw audio to mel
            mel_feats, mel_lens = self.preprocessor(_input, length)
            # Forward encoder với mel
            encoder_out, encoder_out_lens = self.model.forward_encoder(mel_feats, mel_lens)

        if self.ctc_decoder is not None:
            text = self.ctc_beamsearch(encoder_out, encoder_out_lens)
        else:
            ids = self.model.get_predicts(encoder_out, encoder_out_lens)[0]
            text = self.collate_fn.ids2text(ids)
        return text
