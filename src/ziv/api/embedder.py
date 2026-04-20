"""Lightweight ONNX sentence embedder used by Ziv."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)


class LightEmbedder:
    """ONNX-based sentence embedder with mean/CLS pooling support."""

    def __init__(self, model_dir: str, max_length: int = 256):
        """Initialize tokenizer, pooling config, and ONNX session."""
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        self.tokenizer: Tokenizer
        self.session: ort.InferenceSession
        self.pool_mean = True
        self.pool_cls = False
        self.do_normalize = False
        self.use_token_type_ids = False

        self._load_tokenizer()
        self._load_pooling_config()
        self._load_session()

        self._run_opts = ort.RunOptions()
        self._run_opts.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage",
            "cpu:0",
        )

    def _load_tokenizer(self) -> None:
        """Load and configure the tokenizer."""
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {self.model_dir}")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_token="[PAD]", length=None)

    def _load_pooling_config(self) -> None:
        """Load pooling and normalization settings from model metadata."""
        pooling_path = self.model_dir / "1_Pooling" / "config.json"
        if not pooling_path.exists():
            raise FileNotFoundError(
                f"1_Pooling/config.json not found in {self.model_dir}"
            )

        with pooling_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        self.pool_mean = config.get("pooling_mode_mean_tokens", True)
        self.pool_cls = config.get("pooling_mode_cls_token", False)
        self.do_normalize = (self.model_dir / "2_Normalize").is_dir()

    def _load_session(self) -> None:
        """Create the ONNX Runtime session."""
        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"model.onnx not found in {self.model_dir}")

        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        input_names = {
            input_meta.name for input_meta in self.session.get_inputs()}
        self.use_token_type_ids = "token_type_ids" in input_names

        logger.info(
            "LightEmbedder loaded: pool_mean=%s, pool_cls=%s, normalize=%s, token_type_ids=%s",
            self.pool_mean,
            self.pool_cls,
            self.do_normalize,
            self.use_token_type_ids,
        )

    def _tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Convert input texts into ONNX-ready arrays."""
        encoded = self.tokenizer.encode_batch(texts)

        inputs: dict[str, np.ndarray] = {
            "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.asarray(
                [item.attention_mask for item in encoded],
                dtype=np.int64,
            ),
        }

        if self.use_token_type_ids:
            inputs["token_type_ids"] = np.asarray(
                [item.type_ids for item in encoded],
                dtype=np.int64,
            )

        return inputs

    def _mean_pool(
        self,
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply attention-mask-aware mean pooling."""
        mask = attention_mask.astype(np.float32)[..., np.newaxis]
        summed = (token_embeddings * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        return summed / counts

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings in place."""
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True).clip(min=1e-12)
        embeddings /= norms
        return embeddings

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode one or more texts into float32 embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        inputs = self._tokenize(texts)
        outputs = self.session.run(None, inputs, self._run_opts)
        token_embeddings = np.asarray(outputs[0], dtype=np.float32)

        if self.pool_cls:
            embeddings = token_embeddings[:, 0, :]
        else:
            embeddings = self._mean_pool(
                token_embeddings, inputs["attention_mask"])

        if self.do_normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings
