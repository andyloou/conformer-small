from datasets import load_dataset
import sentencepiece as spm
import os

# Bước 1: Load dataset và lưu text
print("Đang load dataset...")
dataset = load_dataset("linhtran92/viet_bud500", split="train", streaming=True)
text_file = "viet_bud500_texts.txt"
if os.path.exists(text_file):
    os.remove(text_file)

with open(text_file, "w", encoding="utf-8") as f:
    for i, item in enumerate(dataset):
        text = item.get("transcription", "").strip()  # Lấy cột 'transcription'
        if text:
            f.write(text + "\n")
        if i % 10000 == 0:
            print(f"Đã xử lý {i} mẫu...")

print(f"Đã lưu {text_file} với transcripts.")

# Bước 2: Train BPE với SentencePiece
print("Đang train BPE với SentencePiece...")
spm.SentencePieceTrainer.train(
    input=text_file,
    model_prefix="bpe",
    vocab_size=2000,
    model_type="bpe",
    character_coverage=0.9995,  # Phù hợp cho tiếng Việt
    user_defined_symbols="[MASK]",  # Thêm token đặc biệt nếu cần
    # Không gán cứng unk_id, bos_id, eos_id, pad_id để tránh xung đột
)
print("Training hoàn tất!")

# Bước 3: Load model để xuất bpe.vocab và word_vocab.txt
sp = spm.SentencePieceProcessor(model_file="bpe.model")

# Xuất bpe.vocab (token và score từ sentencepiece)
with open("bpe.vocab", "w", encoding="utf-8") as f_vocab:
    for i in range(sp.get_piece_size()):
        token = sp.id_to_piece(i)
        score = sp.get_score(i)  # Lấy score (log-probability)
        f_vocab.write(f"{token}\t{score}\n")
print("Đã lưu bpe.vocab")

# Xuất word_vocab.txt (khớp với logic của ASRCollator)
with open("word_vocab.txt", "w", encoding="utf-8") as f_word:
    # Thêm <blank> và <unk> ở đầu
    f_word.write("<blank>\n")
    f_word.write("<unk>\n")
    # Thêm các token từ sentencepiece (bỏ <unk> gốc nếu có)
    for i in range(sp.get_piece_size()):
        token = sp.id_to_piece(i)
        if token != "<unk>":  # Bỏ <unk> gốc vì đã thêm thủ công
            f_word.write(f"{token}\n")
    # Thêm <pad> ở cuối
    f_word.write("<pad>\n")
print("Đã lưu word_vocab.txt")