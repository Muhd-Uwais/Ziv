"""
chunker.py — Sliding-window file content chunker for the ziv indexing pipeline.

This module splits source file content into overlapping line-based chunks
suitable for embedding, semantic search, and retrieval-augmented workflows.
Each chunk is identified by a deterministic content hash so that downstream
consumers can detect and skip unchanged chunks on re-indexing.

Typical usage::

    from ziv.core.chunker import chunk_file, chunk_directory

    # Chunk a single file
    chunks = chunk_file("src/app.py", source_code, chunk_size=40, overlap=10)

    # Chunk an entire directory of pre-loaded file records
    all_chunks = chunk_directory(files_data, chunk_size=40, overlap=10)

Compatibility: Python 3.10+
"""

import logging
from ..utils.hash_utils import compute_hash
from typing import TypedDict


__all__ = ["Chunk", "chunk_file", "chunk_directory"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Chunk(TypedDict):
    """A single line-range chunk extracted from a source file.

    All fields are always present; consumers may rely on complete dicts.

    Attributes:
        id:         Deterministic SHA hash of ``file_path + line_range + content``.
        file_path:  Source file path as provided by the caller.
        start_line: 1-indexed first line of this chunk (inclusive).
        end_line:   1-indexed last line of this chunk (inclusive).
        content:    Raw text of the chunk (newline-joined lines).
    """
    id: str
    file_path: str
    start_line: int
    end_line: int
    content: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_chunk_params(chunk_size: int, overlap: int) -> None:
    """Validate chunking parameters and raise on invalid input.

    Centralising validation here avoids duplicating the same checks across
    every public function that accepts these parameters.

    Args:
        chunk_size: Number of lines per chunk.  Must be a positive ``int``.
        overlap:    Shared lines between consecutive chunks.  Must be a
                    non-negative ``int`` strictly less than *chunk_size*.

    Raises:
        TypeError:  If either argument is not an ``int``.
        ValueError: If *chunk_size* < 1, *overlap* < 0, or *overlap* >=
                    *chunk_size*.
    """
    if not isinstance(chunk_size, int):
        raise TypeError(
            f"chunk_size must be int, got {type(chunk_size).__name__!r}")
    if not isinstance(overlap, int):
        raise TypeError(f"overlap must be int, got {type(overlap).__name__!r}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        # Negative overlap silently creates *gaps* in line coverage —
        # lines would be skipped entirely, corrupting the index.
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    """Split file content into overlapping sliding-window chunks.

    Uses a **line-based sliding window**: each chunk contains at most
    *chunk_size* lines; the window advances by ``chunk_size - overlap`` lines
    per step so that adjacent chunks share *overlap* lines of context.

    Each chunk receives a deterministic hash ID derived from its file path,
    line range, and textual content, making chunks stable across identical
    re-runs and suitable for cache/dedup checks downstream.

    Edge cases handled:

    * **Empty file** — returns ``[]``.
    * **File shorter than** *chunk_size* — returns a single chunk covering
      the whole file.
    * **Partial trailing chunk** — the last window may contain fewer than
      *chunk_size* lines; it is kept as-is so no lines are lost.

    Args:
        file_path:  Path identifying the source file.  Used in hash generation
                    and chunk metadata; the file is *not* read here.
        content:    Full text content of the file to chunk.
        chunk_size: Maximum number of lines per chunk.  Must be >= 1.
                    Defaults to ``40``.
        overlap:    Number of lines shared between consecutive chunks.
                    Must be >= 0 and strictly less than *chunk_size*.
                    Defaults to ``10``.

    Returns:
        Ordered list of :class:`Chunk` dicts.  The list is empty when
        *content* is empty.  Chunks appear in file order; consecutive entries
        overlap by exactly *overlap* lines.

    Raises:
        TypeError:  If *chunk_size* or *overlap* is not an ``int``.
        ValueError: If *chunk_size* < 1, *overlap* < 0, or
                    *overlap* >= *chunk_size*.

    Example::

        >>> chunks = chunk_file("app.py", source_code, chunk_size=30, overlap=5)
        >>> chunks[0]["start_line"], chunks[0]["end_line"]
        (1, 30)
    """
    _validate_chunk_params(chunk_size, overlap)

    lines = content.splitlines()

    # Nothing to chunk — caller may pass an empty or placeholder file.
    if not lines:
        logger.debug("chunk_file: '%s' is empty, returning []", file_path)
        return []

    total_lines = len(lines)

    # --- Fast path: entire file fits within a single chunk ---
    if total_lines <= chunk_size:
        chunk_text = "\n".join(lines)
        chunk_id = compute_hash(f"{file_path}:1:{total_lines}:{chunk_text}")
        logger.debug(
            "chunk_file: '%s' fits in one chunk (%d lines)", file_path, total_lines
        )
        return [
            Chunk(
                id=chunk_id,
                file_path=file_path,
                start_line=1,
                end_line=total_lines,
                content=chunk_text,
            )
        ]

    # --- Sliding-window path ---
    chunks: list[Chunk] = []
    step = chunk_size - overlap  # guaranteed >= 1 after _validate_chunk_params
    start = 0  # 0-indexed current window position

    while start < total_lines:
        end = start + chunk_size
        chunk_lines = lines[start:end]

        # Safety guard: empty slice means we've overshot (should not occur
        # given the loop condition, but avoids an infinite loop if it ever does).
        if not chunk_lines:
            break

        chunk_text = "\n".join(chunk_lines)
        # Clamp end to the real file boundary for metadata accuracy.
        actual_end = min(end, total_lines)

        chunks.append(
            Chunk(
                id=compute_hash(
                    f"{file_path}:{start + 1}:{actual_end}:{chunk_text}"
                ),
                file_path=file_path,
                start_line=start + 1,  # Convert 0-indexed start → 1-indexed
                end_line=actual_end,
                content=chunk_text,
            )
        )

        start += step

    logger.debug(
        "chunk_file: '%s' → %d chunks (chunk_size=%d, overlap=%d)",
        file_path,
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


def chunk_directory(
    files_data: list[dict[str, str]],
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    """Chunk multiple files and return all chunks as a single flat list.

    Iterates over *files_data*, delegates per-file chunking to
    :func:`chunk_file`, and concatenates the results.  Validation of
    *chunk_size* and *overlap* is performed **once** before iterating so
    that callers get a clear error rather than a failure mid-way through a
    large file set.

    Files that produce zero chunks (e.g., empty files) contribute nothing to
    the returned list; no error is raised for them.

    Args:
        files_data: Collection of file records.  Each record must be a
                    ``dict`` containing at minimum:

                    * ``"file_path"`` (:class:`str`) — path to the source file.
                    * ``"content"``   (:class:`str`) — full text of the file.

                    Additional keys in the dict are silently ignored.
        chunk_size: Lines per chunk, forwarded to :func:`chunk_file`.
                    Defaults to ``40``.
        overlap:    Overlap lines, forwarded to :func:`chunk_file`.
                    Defaults to ``10``.

    Returns:
        Flat, ordered list of :class:`Chunk` dicts for all files.  Chunks
        from each file appear in source order; the overall ordering mirrors
        the order of *files_data*.

    Raises:
        TypeError:  If *chunk_size* or *overlap* is not an ``int``.
        ValueError: If *chunk_size* / *overlap* are invalid, or if any
                    record in *files_data* is missing ``"file_path"`` or
                    ``"content"``.

    Example::

        >>> files = [
        ...     {"file_path": "a.py", "content": "x = 1\\n"},
        ...     {"file_path": "b.py", "content": "def foo():\\n    pass\\n"},
        ... ]
        >>> chunks = chunk_directory(files, chunk_size=20, overlap=5)
        >>> len(chunks)
        2
    """
    # Validate once upfront — avoids a confusing mid-iteration error.
    _validate_chunk_params(chunk_size, overlap)

    all_chunks: list[Chunk] = []

    for idx, record in enumerate(files_data):
        # Validate required keys with a helpful error pinpointing the record.
        missing_keys = [k for k in ("file_path", "content") if k not in record]
        if missing_keys:
            raise ValueError(
                f"files_data[{idx}] is missing required key(s): {missing_keys}"
            )

        file_chunks = chunk_file(
            record["file_path"],
            record["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        # extend (O(k)) is faster than += which rebuilds the list reference.
        all_chunks.extend(file_chunks)

    logger.debug(
        "chunk_directory: processed %d file(s) → %d total chunks",
        len(files_data),
        len(all_chunks),
    )
    return all_chunks
