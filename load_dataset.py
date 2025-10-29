import os
from huggingface_hub import snapshot_download
from datasets import load_dataset, DatasetDict


REPO_ID = "linhtran92/viet_bud500"
REPO_TYPE = "dataset"
RAW_DIR = "viet_bud500_raw"
FINAL_DIR = "viet_bud500_processed"

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
print("Bước 1: Đang tải dataset từ Hugging Face...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    local_dir=RAW_DIR,
    resume_download=True,      
    max_workers=1,          
    tqdm_class=None
)
print(f"Đã tải xong vào: {os.path.abspath(RAW_DIR)}")


print("Bước 2: Đang load từ file Parquet...")
data_dir = os.path.join(RAW_DIR, "data")

ds = load_dataset(
    "parquet",
    data_files={
        "train": f"{data_dir}/train-*.parquet",
        "validation": f"{data_dir}/validation-*.parquet",
        "test":  f"{data_dir}/test-*.parquet",
    }
)

print(f"Load thành công!")
print(ds)
print(f"Train: {ds['train'].num_rows:,} mẫu")
print(f"Train: {ds['validation'].num_rows:,} mẫu")
print(f"Test:  {ds['test'].num_rows:,} mẫu")

print(f"Bước 3: Đang lưu dataset đã xử lý vào: {FINAL_DIR}")
ds.save_to_disk(FINAL_DIR)
print("HOÀN TẤT! Dataset đã sẵn sàng để dùng lại.")
print(f"→ Lần sau chỉ cần: load_from_disk('{FINAL_DIR}')")