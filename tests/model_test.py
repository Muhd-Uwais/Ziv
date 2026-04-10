import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

model_dir = ".ziv/models/embedder-fast-onnx"

tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding(pad_token="[PAD]", length=None)

session = ort.InferenceSession(
    os.path.join(model_dir, "model.onnx"),
    providers=["CPUExecutionProvider"]
)

texts = ["How does authentication work?", "Explain the login flow"]
encoded = tokenizer.encode_batch(texts)

input_ids      = np.array([e.ids            for e in encoded], dtype=np.int64)
attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
token_type_ids = np.array([e.type_ids       for e in encoded], dtype=np.int64)

outputs = session.run(None, {
    "input_ids":      input_ids,
    "attention_mask": attention_mask,
    "token_type_ids": token_type_ids,
})

# Mean pooling
token_embeddings = outputs[0]
mask = attention_mask[..., None].astype(np.float32)

embeddings = (token_embeddings * mask).sum(1) / mask.sum(1).clip(min=1e-9)
assert embeddings.shape == (2, 384), f"Expected (2, 384), got {embeddings.shape}"

norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
embeddings = embeddings / norms

sim_matrix = embeddings @ embeddings.T
assert sim_matrix.shape == (2, 2), f"Expected (2,2), got {sim_matrix.shape}"

# Self-similarity should be ~1.0 (L2 normalized)
assert abs(sim_matrix[0, 0] - 1.0) < 1e-5, "Self-cosine should be 1.0"
assert abs(sim_matrix[1, 1] - 1.0) < 1e-5, "Self-cosine should be 1.0"

# The two texts are semantically similar, so expect > 0.7
print(f"Cosine similarity between texts: {sim_matrix[0,1]:.4f}")
print("All assertions passed.")