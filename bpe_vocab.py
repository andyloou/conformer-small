from datasets import load_dataset, load_from_disk
import sentencepiece as spm
import os
import subprocess
import kenlm

# Đường dẫn đến binary KenLM
KENLM_LMPLZ = "/home/andyloou/bud500/viet-asr/kenlm/bin/lmplz"
KENLM_BUILD_BINARY = "/home/andyloou/bud500/viet-asr/kenlm/bin/build_binary"

if not os.path.exists(KENLM_LMPLZ):
    raise FileNotFoundError(f"KenLM binary not found at {KENLM_LMPLZ}. Run: git clone https://github.com/kpu/kenlm.git, build, and copy bin/lmplz.")
if not os.path.exists(KENLM_BUILD_BINARY):
    raise FileNotFoundError(f"KenLM binary not found at {KENLM_BUILD_BINARY}. Run: git clone https://github.com/kpu/kenlm.git, build, and copy bin/build_binary.")

full_dataset_dict = load_from_disk("viet_bud500_processed")
print(f"Đã load các split: {list(full_dataset_dict.keys())}")

text_file = "viet_bud500_texts_FULL.txt"
if os.path.exists(text_file):
    os.remove(text_file)

processed_count = 0
with open(text_file, "w", encoding="utf-8") as f:
    # Lặp qua từng split (train, validation, test)
    for split_name, dataset_split in full_dataset_dict.items():
        print(f"Đang xử lý split: {split_name}...")
        dataset_split = dataset_split.select_columns(["transcription"])

        for item in dataset_split:
            text = item.get("transcription", "").strip()
            if text:
                f.write(text + "\n")
            processed_count += 1
            if processed_count % 20000 == 0:
                print(f"Đã xử lý {processed_count} mẫu (tổng cộng)...")

print(f"Đã lưu {text_file} với {processed_count} transcripts từ tất cả các split.")

# # Bước 2: Train BPE với SentencePiece (vocab 4000)
vocab_size = 4000
print(f"Đang train BPE với SentencePiece (vocab_size={vocab_size})...")
spm.SentencePieceTrainer.train(
    input=text_file, # Phải dùng file text_file ở trên
    model_prefix="data/bpe_4000/bpe",
    vocab_size=vocab_size,
    model_type="bpe",
    character_coverage=0.9995,
    user_defined_symbols="[MASK]",
)
print("Training BPE hoàn tất!")

# Bước 3: Load model để xuất bpe.vocab và word_vocab.txt

print("Loading BPE model...")
sp = spm.SentencePieceProcessor(model_file="data/bpe_4000/bpe.model")

# # Xuất bpe.vocab
with open("data/bpe_4000/bpe.vocab", "w", encoding="utf-8") as f_vocab:
    for i in range(sp.get_piece_size()):
        token = sp.id_to_piece(i)
        score = sp.get_score(i)
        f_vocab.write(f"{token}\t{score}\n")
print("Đã lưu data/bpe_4000/bpe.vocab")
with open("data/bpe_4000/word_vocab.txt", "w", encoding="utf-8") as f_word:
    f_word.write("<blank>\n")
    f_word.write("<unk>\n")
    for i in range(sp.get_piece_size()):
        token = sp.id_to_piece(i)
        if token != "<unk>":
            f_word.write(f"{token}\n")
    f_word.write("<pad>\n")
print("Đã lưu data/bpe_4000/word_vocab.txt")
tokenized_file = "viet_bud500_tokenized_FULL.txt" # Đổi tên file
if os.path.exists(tokenized_file):
    os.remove(tokenized_file)

print(f"Đang tokenizing tất cả các split cho KenLM...")
processed_count = 0
with open(tokenized_file, "w", encoding="utf-8") as f:
    for split_name, dataset_split in full_dataset_dict.items():
        print(f"Đang tokenizing split: {split_name}...")
        dataset_split = dataset_split.select_columns(["transcription"])
        
        for item in dataset_split:
            text = item.get("transcription", "").strip()
            if text:
                tokens = sp.encode(text, out_type=str)
                f.write(" ".join(tokens) + "\n")
            processed_count += 1
            if processed_count % 20000 == 0:
                print(f"Đã tokenize {processed_count} mẫu (tổng cộng)...")
                
print(f"Đã lưu {tokenized_file} với {processed_count} tokenized texts từ tất cả các split.")

# Bước 5: Train KenLM 5-gram với pruning
print("Đang train KenLM 5-gram với pruning...")
arpa_file = "lm_full.arpa"
binary_file = "lm_full.binary"

subprocess.run([
    KENLM_LMPLZ,
    "-o", "5",  # 5-gram
    "--prune", "0", "0", "0", "1",  # Pruning nhẹ (ngưỡng cho 2,3,4,5-grams)
    "--limit_vocab_file", "data/bpe_4000/word_vocab.txt",
    "--text", tokenized_file, # Dùng file _FULL
    "--arpa", arpa_file
], check=True)

# Chuyển ARPA thành binary
print(f"Building binary file {binary_file}...")
subprocess.run([
    KENLM_BUILD_BINARY,
    "trie",  # Dùng trie để tối ưu kích thước
    arpa_file,
    binary_file
], check=True)
print(f"Đã lưu {binary_file}")

# Kiểm tra kích thước file
lm_size = os.path.getsize(binary_file) / (1024 * 1024)  # MB
print(f"Kích thước {binary_file}: {lm_size:.2f} MB")
if lm_size > 50:
    print("Cảnh báo: lm.binary vượt quá 50MB. Thử tăng pruning (--prune 0 2 3) hoặc giảm n-gram order (-o 2).")

# Bước 6: Kiểm tra KenLM model bằng pypi-kenlm
print("Kiểm tra KenLM model...")
try:
    kenlm_model = kenlm.Model(binary_file)
    test_sentence = "xin chào đây là ví dụ"
    tokens = sp.encode(test_sentence, out_type=str)
    tokenized_sentence = " ".join(tokens)
    score = kenlm_model.score(tokenized_sentence, bos=True, eos=True)
    print(f"Điểm KenLM cho câu '{test_sentence}': {score:.4f}")
except Exception as e:
    print(f"Lỗi khi load KenLM model: {e}")

# === SỬA LỖI BƯỚC 7: Kiểm tra coverage (dùng split 'validation' từ dict) ===
print("Kiểm tra coverage trên mẫu validation...")
try:
    val_dataset = full_dataset_dict["validation"]
except KeyError:
    print("Không tìm thấy split 'validation' trong dataset đã load. Bỏ qua kiểm tra coverage.")
    exit() # Thoát nếu không có split validation

val_dataset = val_dataset.select_columns(["transcription"])
unk_count = 0
total_tokens = 0

# .take(1000) không dùng được cho dataset đã load từ disk (non-streaming)
# Cần lặp qua 1000 mẫu đầu tiên
total_val_samples = len(val_dataset)
samples_to_check = min(1000, total_val_samples)

print(f"Kiểm tra {samples_to_check} mẫu từ split validation...")
for i, item in enumerate(val_dataset):
    if i >= samples_to_check:
        break # Chỉ kiểm tra 1000 mẫu đầu
        
    text = item.get("transcription", "").strip()
    if text:
        tokens = sp.encode(text, out_type=str)
        unk_count += tokens.count("<unk>")
        total_tokens += len(tokens)
        
unk_rate = unk_count / total_tokens if total_tokens > 0 else 0
print(f"Tỷ lệ <unk> trên {samples_to_check} mẫu validation: {unk_rate:.4f} ({unk_count}/{total_tokens})")