import glob
import os
import subprocess

import kenlm
import sentencepiece as spm
from datasets import load_from_disk

DATASET_DIR   = "/home/datasets/viet_bud500"
BPE_DIR       = "data/bpe_4000"
VOCAB_SIZE    = 4000
KENLM_ORDER   = 5
KENLM_PRUNE   = ["0", "0", "0", "1"]
LM_SIZE_LIMIT = 50  # MB

RAW_TEXT_FILE       = "viet_bud500_texts.txt"
TOKENIZED_TEXT_FILE = "viet_bud500_tokenized.txt"
ARPA_FILE           = "lm.arpa"
BINARY_FILE         = "lm.binary"

KENLM_LMPLZ        = "/home/andyloou/kenlm/build/bin/lmplz"
KENLM_BUILD_BINARY = "/home/andyloou/kenlm/build/bin/build_binary"

os.makedirs(BPE_DIR, exist_ok=True)


def get_env_with_boost() -> dict:
    env = os.environ.copy()

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    search_patterns = (
        glob.glob(f"{conda_prefix}/lib/libboost_program_options.so*") if conda_prefix else []
    ) + (
        glob.glob(f"{conda_prefix}/pkgs/boost*/lib/libboost_program_options.so*") if conda_prefix else []
    ) + glob.glob("/usr/lib/x86_64-linux-gnu/libboost_program_options.so*") \
      + glob.glob("/usr/lib/*/libboost_program_options.so*") \
      + glob.glob("/usr/local/lib/libboost_program_options.so*")

    dirs = list(dict.fromkeys(os.path.dirname(p) for p in search_patterns if p))
    if dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(dirs) + (f":{existing}" if existing else "")

    return env


def iter_texts(split: str):
    ds = load_from_disk(DATASET_DIR)[split].select_columns(["transcription"])
    for item in ds:
        text = item.get("transcription", "").strip()
        if text:
            yield text


def write_texts_to_file(path: str):
    with open(path, "w", encoding="utf-8") as f:
        for text in iter_texts("train"):
            f.write(text + "\n")
    print(f"Saved: {path}")


def train_bpe(text_file: str):
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=f"{BPE_DIR}/bpe",
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        user_defined_symbols="[MASK]",
    )


def export_vocab(sp: spm.SentencePieceProcessor):
    with open(f"{BPE_DIR}/bpe.vocab", "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
    print(f"Saved: {BPE_DIR}/bpe.vocab")

    with open(f"{BPE_DIR}/word_vocab.txt", "w", encoding="utf-8") as f:
        f.write("<blank>\n<unk>\n")
        for i in range(sp.get_piece_size()):
            token = sp.id_to_piece(i)
            if token != "<unk>":
                f.write(f"{token}\n")
        f.write("<pad>\n")
    print(f"Saved: {BPE_DIR}/word_vocab.txt")


def tokenize_corpus(sp: spm.SentencePieceProcessor, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for text in iter_texts("train"):
            tokens = sp.encode(text, out_type=str)
            f.write(" ".join(tokens) + "\n")
    print(f"Saved: {path}")


def train_kenlm():
    env = get_env_with_boost()
    subprocess.run(
        [
            KENLM_LMPLZ,
            "-o", str(KENLM_ORDER),
            "--prune", *KENLM_PRUNE,
            "--limit_vocab_file", f"{BPE_DIR}/word_vocab.txt",
            "--text", TOKENIZED_TEXT_FILE,
            "--arpa", ARPA_FILE,
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [KENLM_BUILD_BINARY, "trie", ARPA_FILE, BINARY_FILE],
        env=env,
        check=True,
    )

    size_mb = os.path.getsize(BINARY_FILE) / 1024 ** 2
    print(f"Saved: {BINARY_FILE} ({size_mb:.1f} MB)")
    if size_mb > LM_SIZE_LIMIT:
        print(f"Warning: {BINARY_FILE} exceeds {LM_SIZE_LIMIT} MB, consider increasing --prune or reducing -o")


def verify_kenlm(sp: spm.SentencePieceProcessor):
    model = kenlm.Model(BINARY_FILE)
    test = "xin chào đây là ví dụ"
    score = model.score(" ".join(sp.encode(test, out_type=str)), bos=True, eos=True)
    print(f"KenLM score for '{test}': {score:.4f}")


def check_unk_rate(sp: spm.SentencePieceProcessor, n_samples: int = 1_000):
    unk, total = 0, 0
    ds = load_from_disk(DATASET_DIR)["validation"].select_columns(["transcription"])
    for item in ds.select(range(min(n_samples, len(ds)))):
        text = item.get("transcription", "").strip()
        if text:
            tokens = sp.encode(text, out_type=str)
            unk   += tokens.count("<unk>")
            total += len(tokens)
    rate = unk / total if total else 0
    print(f"<unk> rate on validation: {rate:.4f} ({unk}/{total})")


def main():
    write_texts_to_file(RAW_TEXT_FILE)
    train_bpe(RAW_TEXT_FILE)

    sp = spm.SentencePieceProcessor(model_file=f"{BPE_DIR}/bpe.model")
    export_vocab(sp)
    tokenize_corpus(sp, TOKENIZED_TEXT_FILE)
    train_kenlm()
    verify_kenlm(sp)
    check_unk_rate(sp)


if __name__ == "__main__":
    main()