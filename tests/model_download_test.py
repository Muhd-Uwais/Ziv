import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

REPO_ID = "ziv-ai/embedder-fast-onnx"
LOCAL_DIR = ".ziv/models/embedder-fast-onnx"

# Required files that must exist after download
REQUIRED_FILES = [
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
]

def test_snapshot_download():
    # Step 1: Download
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )

    # Step 2: Directory exists
    assert os.path.isdir(LOCAL_DIR), \
        f"Download directory not created: {LOCAL_DIR}"

    # Step 3: All required files exist and are non-empty
    for filename in REQUIRED_FILES:
        filepath = os.path.join(LOCAL_DIR, filename)
        assert os.path.isfile(filepath), \
            f"Missing required file: {filename}"
        assert os.path.getsize(filepath) > 0, \
            f"File is empty: {filename}"

    # Step 4: model.onnx is a reasonable size (> 10MB)
    model_path = os.path.join(LOCAL_DIR, "model.onnx")
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    assert model_size_mb > 10, \
        f"model.onnx too small ({model_size_mb:.1f}MB) — likely corrupted"

    print(f"model.onnx size: {model_size_mb:.1f}MB")
    print("All download assertions passed.")

if __name__ == "__main__":
    test_snapshot_download()