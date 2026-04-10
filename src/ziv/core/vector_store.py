"""vector_store.py — FAISS index lifecycle management for the lift search pipeline.

Owns the three operations every retrieval workflow needs:

  1. **Build** — convert raw float embeddings into a FAISS ``IndexFlatIP``,
     L2-normalise them, and persist both the index binary and the integer→
     metadata map to ``OUTPUT_DIR``.
  2. **Load** — restore a previously built index and its companion ID map from
     disk, transparently converting JSON string keys back to ``int``.
  3. **Search** — run a normalised nearest-neighbour query against a loaded
     index and return enriched result dicts ready for the CLI output layer.

``IndexFlatIP`` on L2-normalised vectors is equivalent to cosine similarity.
Scores therefore lie in ``[0.0, 1.0]`` for well-formed inputs; callers may
rely on that invariant.

All path arguments default to ``OUTPUT_DIR`` (``.ziv/``), the only directory
lift writes to at runtime.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Filenames are module-level constants so every module that references these
# paths imports from one place rather than scattering string literals.
INDEX_FILENAME = "index.faiss"
ID_MAP_FILENAME = "id_map.json"
OUTPUT_DIR = ".ziv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_and_save(
    embeddings: list[list[float]],
    metadata: list[dict],
    output_dir: str | Path = OUTPUT_DIR,
) -> None:
    """Build an ``IndexFlatIP`` from *embeddings* and persist it to *output_dir*.

    Called by the index-builder stage after embedding generation completes.
    Vectors are L2-normalised before insertion even when the upstream embedder
    already normalises them — a hard guarantee is cheaper than a silent
    correctness assumption that breaks if the embedder ever changes.

    **ID map** — FAISS returns integer row positions, not the original chunk
    identifiers.  A ``{int → chunk_metadata_dict}`` companion map is therefore
    serialised alongside the index.  Position ``i`` in the map corresponds to
    row ``i`` in the vector matrix; the alignment is enforced by the
    ``len`` equality check below.

    Raises:
        ValueError: If *embeddings* is empty, or if *embeddings* and *metadata*
            have different lengths.  A length mismatch would produce a corrupted
            index where some FAISS row positions have no metadata entry,
            surfacing as a silent ``None`` return in ``search`` rather than a
            loud failure here.
    """
    if not embeddings:
        raise ValueError("Cannot build index from empty embeddings list")

    # Enforce alignment before doing any work — a mismatch here means the
    # caller's pipeline is broken, and we want a loud error at the source.
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"embeddings and metadata must have the same length "
            f"(got {len(embeddings)} vs {len(metadata)})"
        )

    vectors = np.array(embeddings, dtype=np.float32)
    dim = vectors.shape[1]

    # Normalise before insertion — idempotent on unit vectors, protective
    # on anything the caller forgot to normalise.
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # FAISS I/O functions expect a plain string, not a Path object.
    faiss.write_index(index, str(out / INDEX_FILENAME))

    # integer FAISS row position → chunk metadata.
    # Index i aligns with row i in the vectors array.
    id_map = {i: meta for i, meta in enumerate(metadata)}

    with open(out / ID_MAP_FILENAME, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=4)

    logger.info(
        "FAISS index built: %d vectors, dim=%d → saved to '%s'",
        len(embeddings),
        dim,
        out,
    )


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
    id_map: dict[int, dict],
    query_vector: np.ndarray | list[float],
    k: int = 5,
) -> list[dict]:
    """Return the top-*k* nearest chunks to *query_vector*.

    **Index type** — the parameter is typed as ``faiss.Index`` (the base
    class) rather than ``faiss.IndexFlatIP`` so callers can swap in any FAISS
    index (``IndexHNSWFlat``, ``IndexIVFFlat``, …) without a type error.

    **``np.asarray`` vs ``np.array``** — ``asarray`` avoids an unnecessary
    copy when *query_vector* is already a correctly-typed ``ndarray``; it
    returns a view in that case.  The reshape and normalise operations are
    still safe because FAISS does not hold a reference to the query buffer
    after ``search`` returns.

    **Normalisation** — the query is normalised unconditionally.  Scores are
    therefore cosine similarities in ``[0.0, 1.0]``; the caller can treat them
    as such without knowing how the query was produced.

    **FAISS ``-1`` padding** — when fewer than *k* results exist, FAISS pads
    the output arrays with ``-1``.  Those sentinels are silently dropped so
    the caller receives a shorter list rather than an error.

    **Orphaned indices** — a FAISS position with no ``id_map`` entry (possible
    if the index and map files are out of sync) is logged as a warning and
    skipped.  A partial result is preferable to crashing an interactive
    search session.

    Raises:
        ValueError: If *k* < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # asarray avoids a copy when query_vector is already a float32 ndarray.
    query = np.asarray(query_vector, dtype=np.float32)

    if query.ndim == 1:
        # FAISS expects a 2-D matrix of shape (n_queries, dim).
        query = query.reshape(1, -1)

    # Unconditional normalisation — do not trust the caller to have done it.
    faiss.normalize_L2(query)

    # Guard: cannot request more neighbours than vectors in the index.
    k = min(k, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(query, k)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS sentinel — fewer than k results exist.
            continue

        chunk = id_map.get(int(idx))
        if chunk is None:
            # Index and map are out of sync — recoverable, but worth surfacing.
            logger.warning(
                "FAISS returned index %d with no metadata entry — "
                "index and id_map may be out of sync; skipping",
                idx,
            )
            continue

        results.append({
            "score":      float(score),
            "file_path":  chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line":   chunk["end_line"],
            "content":    chunk["content"],
            "chunk_id":   chunk["id"],
        })

    return results


def is_index_built(output_dir: str | Path = OUTPUT_DIR) -> bool:
    """Return True if both index artefacts exist in *output_dir*.

    Intended as a cheap pre-flight check in the retriever so it can surface a
    clean user-facing error before attempting a ``load()`` that would raise
    ``FileNotFoundError`` deep inside the pipeline.
    """
    out = Path(output_dir)
    return (out / INDEX_FILENAME).exists() and (out / ID_MAP_FILENAME).exists()