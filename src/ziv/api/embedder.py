import os
import json
import logging
import numpy as np
import psutil
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import Union


logger = logging.getLogger(__name__)


class LightEmbeddder:
    """
    Lightweight ONNX-based sentence embedder.
    """

    def __init__(self, model_dir: str, max_length: int = 256):

        self.model_dir = model_dir
        self.max_length = max_length
        self._load(model_dir, max_length)
        self._run_opts = ort.RunOptions()
        self._run_opts.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage", "cpu:0")

    def _load(self, model_dir: str, max_length: int):

        # Load tokenizer from tokenizer.json
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

        self.tokenizer = Tokenizer.from_file(
            os.path.join(model_dir, "tokenizer.json")
        )
        # Tokenizer.from_file reads the raw tokenizer.json directly
        # This is the same file Sentence Transformers uses — no format change

        self.tokenizer.enable_truncation(max_length=max_length)
        # if a sentence is longer than max_length tokens, cut it off

        self.tokenizer.enable_padding(pad_token="[PAD]", length=None)
        # when encoding a batch of sentences, shorter ones get padded with [PAD]

        # Read pooling mode from 1_Pooling/config.json
        pooling_path = os.path.join(model_dir, "1_Pooling", "config.json")
        if not os.path.exists(pooling_path):
            raise FileNotFoundError(
                f"1_Pooling/config.json not found in {model_dir}")

        with open(pooling_path) as f:
            pcfg = json.load(f)

        self.pool_mean = pcfg.get("pooling_mode_mean_tokens", True)
        self.pool_cls = pcfg.get("pooling_mode_cls_token", False)
        # we read the JSON and store which pooling mode the model uses
        # for all-MiniLM-L6-v2 this will be pool_mean=True, pool_cls=False

        # Check if 2_Normalize exists -> do we normalize?
        self.do_normalize = os.path.isdir(
            os.path.join(model_dir, "2_Normalize")
        )
        # if the folder exists (even if empty), we L2-normalize the output

        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"model.onnx not found in {model_dir}")

        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=False)
        sess_options.inter_op_num_threads = 1

        # Load the ONNX model into onnxruntime
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        # InferenceSession loads and compiles the ONNX graph for the CPU
        # providers list can include "CUDAExecutionProvider" for GPU

        input_names = {inp.name for inp in self.session.get_inputs()}
        self.use_token_type_ids = "token_type_ids" in input_names
        # If the exported model was from BERT → token_type_ids is present
        # If from RoBERTa → it won't be, and we skip it
        # This makes the class work for any model without code changes

        logger.info(
            "LightEmbedder loaded: pool_mean=%s, normalize=%s, token_type_ids=%s",
            self.pool_mean, self.do_normalize, self.use_token_type_ids
        )

    # Tokenize a batch of sentences
    def _tokenize(self, texts: list) -> tuple:

        encoded = self.tokenizer.encode_batch(texts)

        inputs = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        # attention_mask: 1 for real tokens, 0 for padding tokens
        # We use this in mean pooling to ignore the padding positions

        if self.use_token_type_ids:
            inputs["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        # dtype must be int64 — this matches what we declared in input_names during export

        return inputs

    def _mean_pool(self, token_embeddings, attention_mask) -> np.ndarray:

        mask = attention_mask.astype(np.float32)[:, :, np.newaxis]

        summed = (token_embeddings * mask).sum(axis=1)

        # Sum real token vectors per sentence -> (batch, hidden_dim)

        counts = mask.sum(axis=1).clip(min=1e-9)

        return summed / counts
        # Average of real token vectors = sentence embedding → (batch, hidden_dim)

    def _l2_normalize(self, embeddings) -> np.ndarray:
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True).clip(min=1e-12)
        embeddings /= norms
        return embeddings

    def encode(self, texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self._tokenize(texts)

        outputs = self.session.run(None, inputs, self._run_opts)
        # Run the ONNX graph
        # None = return all outputs
        # outputs[0] = last_hidden_state: (batch, seq_len, hidden_dim)

        token_embeddings = outputs[0]

        if self.pool_mean:
            embeddings = self._mean_pool(
                token_embeddings, inputs["attention_mask"])
        elif self.pool_cls:
            embeddings = token_embeddings[:, 0, :]
            # CLS token is always at position 0
        else:
            embeddings = self._mean_pool(
                token_embeddings, inputs["attention_mask"])

        if self.do_normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings
        # shape: (num_sentences, hidden_dim)
