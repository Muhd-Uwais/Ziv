# scripts/convert_to_onnx.py
#
# DEV ONLY — not part of the ziv package, not installed by pip
#
# Converts a Sentence Transformers model.safetensors to model.onnx
# Run once locally before uploading the ONNX repo to Hugging Face.
#
# Usage:
#   python scripts/convert_to_onnx.py

import os
import shutil
import torch
import onnx
from transformers import AutoTokenizer, AutoModel


# ----- PATHS ------
src = "/home/nox/Codes/ziv/.ziv/models/all-MiniLM-L6-v2"       # Use Path module file not hardcoded
dst = "/home/nox/Codes/ziv/.ziv/models/all-MiniLM-L6-v2-onnx"  # Use Path file not hardcoded
os.makedirs(dst, exist_ok=True)

# STEP 1: Load the PyTorch model from safetensors

model = AutoModel.from_pretrained(src)

model.eval()
# .eval() switches off dropout layers (used during training for regularization)
# during inference we want deterministic, stable outputs — dropout must be off

tokenizer = AutoTokenizer.from_pretrained(src)
# loads tokenizer.json + tokenizer_config.json
# we only need this to create a dummy input for the export — not for real inference


# STEP 2: Create a dummy input

dummy = tokenizer(
    "This is a dummy sentence for export",
    return_tensors="pt",
    padding=True,
)
# ONNX export works by tracing: it runs the model once with this dummy input
# and records every operation. The actual text content doesn't matter,
# only the shape and dtype of the tensors matter.

# STEP 3: Export to ONNX

onnx_path = os.path.join(dst, "model.onnx")

torch.onnx.export(
    model,
    (
        dummy["input_ids"],
        dummy["attention_mask"],
        dummy["token_type_ids"],
    ),
    onnx_path,
    input_names=[
        "input_ids",
        "attention_mask",
        "token_type_ids"
    ],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
    opset_version=18,
    # opset = ONNX operator set version
)

print(f"Exported: {onnx_path}")

# STEP 4: Merge onnx and data.onnx into one

model = onnx.load(onnx_path)

onnx.save(
    model,
    onnx_path,
    save_as_external_data=False
)

os.remove("/home/nox/Codes/ziv/.ziv/models/all-MiniLM-L6-v2-onnx/model.onnx.data") # Use Path unlink()

# STEP 5: Copy all other files across -------

files_to_copy = [
    "tokenizer.json",
    "tokenizer_config.json",
    "sentence_bert_config.json",
    "modules.json",
    "config.json"
]

for fname in files_to_copy:
    src_file = os.path.join(src, fname)
    if os.path.exists(src_file):
        shutil.copy(src_file, dst)
        print(f"Copied: {fname}")


# Copy 1_Pooling folder
shutil.copytree(
    os.path.join(src, "1_Pooling"),
    os.path.join(dst, "1_Pooling"),
    dirs_exist_ok=True
)

# Copy 2_Normalize
shutil.copytree(
    os.path.join(src, "2_Normalize"),
    os.path.join(dst, "2_Normalize"),
    dirs_exist_ok=True
)

print("Done. You can now uninstall torch.")
