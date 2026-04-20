"""vector_store.py — FAISS index lifecycle management for the Ziv search pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np


logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
ID_MAP_FILENAME = "id_map.json"
OUTPUT_DIR = ".ziv"


def _as_float32_matrix(embeddings: list[list[float]] | np.ndarray) -> np.ndarray:
    """Return a 2D float32 matrix."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D matrix")
    return vectors


def build_and_save(
    embeddings: list[list[float]] | np.ndarray,
    metadata: list[dict[str, Any]],
    output_dir: str | Path = OUTPUT_DIR,
) -> None:
    """Build a FAISS IndexFlatIP index and save it with its metadata map."""
    vectors = _as_float32_matrix(embeddings)

    if len(vectors) != len(metadata):
        raise ValueError(
            f"embeddings and metadata must have the same length "
            f"(got {len(vectors)} vs {len(metadata)})"
        )

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / INDEX_FILENAME))

    id_map = {i: meta for i, meta in enumerate(metadata)}
    (out / ID_MAP_FILENAME).write_text(json.dumps(id_map, indent=2), encoding="utf-8")

    logger.info("FAISS index built: %d vectors, dim=%d -> %s",
                len(vectors), vectors.shape[1], out)


def load(output_dir: str | Path = OUTPUT_DIR) -> tuple[faiss.Index, dict[int, dict[str, Any]]]:
    """Load the FAISS index and its metadata map from disk."""
    out = Path(output_dir)
    index_path = out / INDEX_FILENAME
    id_map_path = out / ID_MAP_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. Run `ziv build-index` first.")
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"No ID map found at '{id_map_path}'. Run `ziv build-index` first.")

    index = faiss.read_index(str(index_path))
    raw = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map = {int(k): v for k, v in raw.items()}

    logger.info("FAISS index loaded: %d vectors from '%s'", index.ntotal, out)
    return index, id_map


def load(
    output_dir: str | Path = OUTPUT_DIR,
) -> tuple[faiss.Index, dict[int, dict]]:
    """Restore a previously built index and its ID map from *output_dir*.

    **JSON key coercion** — ``json.dump`` serialises integer dict keys as
    strings (the JSON spec only allows string object keys).  The returned
    ``id_map`` converts them back to ``int`` so callers can index it directly
    with the integer positions FAISS returns, without knowing about the
    serialisation detail.

    Raises:
        FileNotFoundError: If either artefact is absent, with an actionable
            message pointing to the CLI command that creates them.
    """
    out = Path(output_dir)
    index_path = out / INDEX_FILENAME
    id_map_path = out / ID_MAP_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. Run `ziv build-index` first."
        )
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"No ID map found at '{id_map_path}'. Run `ziv build-index` first."
        )

    # FAISS I/O functions expect a plain string, not a Path object.
    index = faiss.read_index(str(index_path))

    with open(id_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Restore int keys — JSON round-trips stringify all dict keys.
    id_map: dict[int, dict] = {int(k): v for k, v in raw.items()}

    logger.info(
        "FAISS index loaded: %d vectors from '%s'",
        index.ntotal,
        out,
    )
    return index, id_map


def search(
    index: faiss.Index,
    id_map: dict[int, dict[str, Any]],
    query_vector: np.ndarray | list[float],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-k nearest chunks for a query vector."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1:
        raise ValueError(
            "query_vector must be a 1D vector or a single-row matrix")

    faiss.normalize_L2(query)

    k = min(k, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(query, k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = id_map.get(int(idx))
        if chunk is None:
            logger.warning(
                "FAISS returned index %d with no metadata entry; skipping", idx)
            continue

        results.append(
            {
                "score": float(score),
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content": chunk["content"],
                "chunk_id": chunk["id"],
            }
        )

    return results


def is_index_built(output_dir: str | Path = OUTPUT_DIR) -> bool:
    """Return True if both persisted FAISS artefacts exist."""
    out = Path(output_dir)
    return (out / INDEX_FILENAME).exists() and (out / ID_MAP_FILENAME).exists()
