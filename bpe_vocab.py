from datasets import load_dataset
import sentencepiece as spm
import os
import subprocess
import kenlm

# Đường dẫn đến binary KenLM
KENLM_LMPLZ = "/home/andyloou/bud500/viet-asr/kenlm/bin/lmplz"
KENLM_BUILD_BINARY = "/home/andyloou/bud500/viet-asr/kenlm/bin/build_binary"

# Kiểm tra binary tồn tại
if not os.path.exists(KENLM_LMPLZ):
    raise FileNotFoundError(f"KenLM binary not found at {KENLM_LMPLZ}. Run: git clone https://github.com/kpu/kenlm.git, build, and copy bin/lmplz.")
if not os.path.exists(KENLM_BUILD_BINARY):
    raise FileNotFoundError(f"KenLM binary not found at {KENLM_BUILD_BINARY}. Run: git clone https://github.com/kpu/kenlm.git, build, and copy bin/build_binary.")

# Bước 1: Load dataset, chỉ lấy cột transcription
print("Đang load dataset...")
dataset = load_dataset("linhtran92/viet_bud500", split="train", streaming=True, keep_in_memory=False)
dataset = dataset.select_columns(["transcription"])

text_file = "viet_bud500_texts.txt"
# if os.path.exists(text_file):
#     os.remove(text_file)

# with open(text_file, "w", encoding="utf-8") as f:
#     for i, item in enumerate(dataset):
#         text = item.get("transcription", "").strip()
#         if text:
#             f.write(text + "\n")
#         if i % 10000 == 0:
#             print(f"Đã xử lý {i} mẫu...")

# print(f"Đã lưu {text_file} với transcripts.")

# # Bước 2: Train BPE với SentencePiece (vocab 4000)
# vocab_size = 4000
# print(f"Đang train BPE với SentencePiece (vocab_size={vocab_size})...")
# spm.SentencePieceTrainer.train(
#     input=text_file,
#     model_prefix="data/bpe_4000/bpe",
#     vocab_size=vocab_size,
#     model_type="bpe",
#     character_coverage=0.9995,
#     user_defined_symbols="[MASK]",
# )
# print("Training BPE hoàn tất!")

# # Bước 3: Load model để xuất bpe.vocab và word_vocab.txt
sp = spm.SentencePieceProcessor(model_file="data/bpe_4000/bpe.model")

# # Xuất bpe.vocab
# with open("data/bpe_4000/bpe.vocab", "w", encoding="utf-8") as f_vocab:
#     for i in range(sp.get_piece_size()):
#         token = sp.id_to_piece(i)
#         score = sp.get_score(i)
#         f_vocab.write(f"{token}\t{score}\n")
# print("Đã lưu data/bpe_4000/bpe.vocab")

# # Xuất word_vocab.txt (khớp ASRCollator)
# with open("data/bpe_4000/word_vocab.txt", "w", encoding="utf-8") as f_word:
#     f_word.write("<blank>\n")
#     f_word.write("<unk>\n")
#     for i in range(sp.get_piece_size()):
#         token = sp.id_to_piece(i)
#         if token != "<unk>":
#             f_word.write(f"{token}\n")
#     f_word.write("<pad>\n")
# print("Đã lưu data/bpe_4000/word_vocab.txt")

# # Bước 4: Tokenize text cho KenLM
# tokenized_file = "viet_bud500_tokenized.txt"
# if os.path.exists(tokenized_file):
#     os.remove(tokenized_file)

# with open(tokenized_file, "w", encoding="utf-8") as f:
#     dataset = load_dataset("linhtran92/viet_bud500", split="train", streaming=True)
#     dataset = dataset.select_columns(["transcription"])
#     for i, item in enumerate(dataset):
#         text = item.get("transcription", "").strip()
#         if text:
#             tokens = sp.encode(text, out_type=str)
#             f.write(" ".join(tokens) + "\n")
#         if i % 10000 == 0:
#             print(f"Đã tokenize {i} mẫu...")
# print(f"Đã lưu {tokenized_file} với tokenized texts.")
tokenized_file = "viet_bud500_tokenized.txt"
# Bước 5: Train KenLM 3-gram với pruning
print("Đang train KenLM 3-gram với pruning...")
subprocess.run([
    KENLM_LMPLZ,
    "-o", "5",  # 3-gram
     "--prune", "0", "0","0","1",  # Pruning nhẹ
    "--limit_vocab_file", "data/bpe_4000/word_vocab.txt",
    "--text", tokenized_file,
    "--arpa", "lm.arpa"
], check=True)

# Chuyển ARPA thành binary
subprocess.run([
    KENLM_BUILD_BINARY,
    "trie",  # Dùng trie để tối ưu kích thước
    "lm.arpa",
    "lm.binary"
], check=True)
print("Đã lưu lm.binary")

# Kiểm tra kích thước file
lm_size = os.path.getsize("lm.binary") / (1024 * 1024)  # MB
print(f"Kích thước lm.binary: {lm_size:.2f} MB")
if lm_size > 50:
    print("Cảnh báo: lm.binary vượt quá 50MB. Thử tăng pruning (--prune 0 2 3) hoặc giảm n-gram order (-o 2).")

# Bước 6: Kiểm tra KenLM model bằng pypi-kenlm
print("Kiểm tra KenLM model...")
try:
    kenlm_model = kenlm.Model("lm.binary")
    test_sentence = "xin chào đây là ví dụ"
    tokens = sp.encode(test_sentence, out_type=str)
    tokenized_sentence = " ".join(tokens)
    score = kenlm_model.score(tokenized_sentence, bos=True, eos=True)
    print(f"Điểm KenLM cho câu '{test_sentence}': {score:.4f}")
except Exception as e:
    print(f"Lỗi khi load KenLM model: {e}")

# Bước 7: Kiểm tra coverage
print("Kiểm tra coverage trên mẫu validation...")
val_dataset = load_dataset("linhtran92/viet_bud500", split="validation", streaming=True)
val_dataset = val_dataset.select_columns(["transcription"])
unk_count = 0
total_tokens = 0
for item in val_dataset.take(1000):
    text = item.get("transcription", "").strip()
    if text:
        tokens = sp.encode(text, out_type=str)
        unk_count += tokens.count("<unk>")
        total_tokens += len(tokens)
unk_rate = unk_count / total_tokens if total_tokens > 0 else 0
print(f"Tỷ lệ <unk> trên validation: {unk_rate:.4f} ({unk_count}/{total_tokens})")