# test_model.py updated (input giờ là mel feats [B, D=80, T=100], mel_lens)
import torch
from vietasr.model import ConformerCTC as ASRModel  # Hoặc from model import ...
from utils import load_config

# Load config
config = load_config("config/conformer.yaml")
print("Config loaded:", config["model"])

# Init model
vocab_size = 2000
pad_id = 0
model = ASRModel(vocab_size=vocab_size, pad_id=pad_id, **config["model"])
model.eval()
print("\n=== MODEL INIT SUCCESS ===")
print(model)

# Manual param count
print("\n=== MODEL SUMMARY (MANUAL PARAM COUNT) ===")
total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())
spec_aug_params = sum(p.numel() for p in model.spec_augmentation.parameters())
print(f"Total params: {total_params:,}")
print(f"Encoder params: {encoder_params:,} (~{encoder_params/total_params*100:.1f}%)")
print(f"Decoder params: {decoder_params:,} (~{decoder_params/total_params*100:.1f}%)")
print(f"SpecAug params: {spec_aug_params:,} (~{spec_aug_params/total_params*100:.1f}%)")

# Dummy mel inputs [B, D=80, T=100]
mel_feats = torch.randn((4, 80, 100))  # [B, mel_bins=80, T_frames=100]
mel_lens = torch.LongTensor([100, 80, 90, 50])  # [B], max=100
target = torch.randint(0, vocab_size, (4, 4))  # [B, max_target_len=4]
target_lens = torch.LongTensor([1, 2, 3, 4])

print("\n=== FORWARD PASS TEST (NO LOAD) ===")
with torch.no_grad():
    out = model(mel_feats, mel_lens, target, target_lens)
    print("Output keys:", list(out.keys()))
    print("Loss:", out["loss"].item())
    print("CTC Loss:", out["ctc_loss"].item())
    print("Decoder Loss:", out["decoder_loss"].item())
    print("Encoder out shape:", out["encoder_out"].shape)  # [4, ~25, 176] (sau sub=4x, T/4)
    print("Log probs shape:", out["log_probs"].shape)  # [4, ~25, 2001]
    print("No NaN/Inf:", not torch.isnan(out["loss"]).any())

# Selective load (giữ nguyên)
pretrained_path = "/home/andyloou/bud500/viet-asr/model_weights.pth"  # Giữ path của bạn
print(f"\n=== SELECTIVE LOAD ENCODER FROM {pretrained_path} ===")
encoder_weight_before = model.encoder.layers[0].self_attn.linear_q.weight.data.clone()
print("Encoder weight before (mean):", encoder_weight_before.mean().item())
model.load_checkpoint(pretrained_path)
encoder_weight_after = model.encoder.layers[0].self_attn.linear_q.weight.data.clone()
print("Encoder weight after (mean):", encoder_weight_after.mean().item())
print("Weight changed?", not torch.equal(encoder_weight_before, encoder_weight_after))

# Forward sau load
print("\n=== FORWARD PASS TEST (AFTER LOAD) ===")
with torch.no_grad():
    out_loaded = model(mel_feats, mel_lens, target, target_lens)
    print("Loss after load:", out_loaded["loss"].item())
    print("Loss improved?", out_loaded["loss"].item() < out["loss"].item())

# Inference (no targets)
print("\n=== INFERENCE TEST (NO TARGETS) ===")
with torch.no_grad():
    inf_out = model(mel_feats, mel_lens)
    print("Inference keys:", list(inf_out.keys()))
    print("Encoder out shape:", inf_out["encoder_out"].shape)
    print("Log probs shape:", inf_out["log_probs"].shape)

# Greedy decode
print("\n=== GREEDY DECODE TEST ===")
with torch.no_grad():
    enc_out, enc_lens = model.forward_encoder(mel_feats, mel_lens)
    print("Forward_encoder:", enc_out.shape, enc_lens)
    predicts = model.get_predicts(enc_out, enc_lens)
    print("Predicted IDs (first 3 samples, top 10):")
    for i in range(min(3, len(predicts))):
        pred_ids = predicts[i].tolist()
        print(f"  Sample {i}: {pred_ids[:10]}... (len={len(pred_ids)})")

print("\n=== TEST COMPLETE ===")
print("SUCCESS if no errors, finite loss, encoder loaded (weight changed).")