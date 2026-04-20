"""
chunker.py — Sliding-window line-based chunker for the ziv indexing pipeline.
"""

from typing import TypedDict, Any
import logging

from ..utils.hash_utils import compute_hash


__all__ = ["Chunk", "chunk_file", "chunk_directory"]


logger = logging.getLogger(__name__)


class Chunk(TypedDict):
    """
    A line‑range chunk from a source file.

    Fields:
      id:        Deterministic hash of file_path + start_line + end_line + content.
      file_path: Source path (not read here).
      start_line: 1‑indexed start line (inclusive).
      end_line:   1‑indexed end line (inclusive).
      content:    Line‑joined text of the chunk.
    """
    id: str
    file_path: str
    start_line: int
    end_line: int
    content: str


def _validate_chunk_params(chunk_size: int, overlap: int) -> None:
    """Raise on invalid chunk_size / overlap."""
    if not isinstance(chunk_size, int):
        raise TypeError(
            f"chunk_size must be int, got {type(chunk_size).__name__!r}")
    if not isinstance(overlap, int):
        raise TypeError(f"overlap must be int, got {type(overlap).__name__!r}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
        )


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    """
    Split file content into overlapping line‑based chunks.

    Uses a sliding window of `chunk_size` lines with `overlap` shared lines.
    Each chunk gets a deterministic hash ID from path, range, and content.

    Args:
      file_path: Source file path.
      content:   Full file text (already loaded).
      chunk_size: Max lines per chunk; ≥ 1, default 40.
      overlap:    Overlap lines between chunks; 0 ≤ overlap < chunk_size, default 10.

    Returns:
      List of `Chunk` records in file order, or `[]` if content is empty.

    Raises:
      TypeError: If chunk_size or overlap is not int.
      ValueError: If chunk_size < 1 or overlap is out of bounds.
    """
    _validate_chunk_params(chunk_size, overlap)

    lines = content.splitlines()
    if not lines:
        logger.debug("chunk_file: '%s' is empty", file_path)
        return []

    total_lines = len(lines)

    if total_lines <= chunk_size:
        chunk_text = "\n".join(lines)
        chunk_id = compute_hash(f"{file_path}:1:{total_lines}:{chunk_text}")
        logger.debug("chunk_file: '%s' fits in one chunk (%d lines)",
                     file_path, total_lines)
        return [Chunk(
            id=chunk_id,
            file_path=file_path,
            start_line=1,
            end_line=total_lines,
            content=chunk_text,
        )]

    chunks: list[Chunk] = []
    step = chunk_size - overlap  # ≥ 1
    start = 0  # 0‑indexed

    while start < total_lines:
        end = start + chunk_size
        chunk_lines = lines[start:end]
        if not chunk_lines:  # safety; should not occur
            break

        chunk_text = "\n".join(chunk_lines)
        actual_end = min(end, total_lines)

        chunks.append(Chunk(
            id=compute_hash(
                f"{file_path}:{start + 1}:{actual_end}:{chunk_text}"),
            file_path=file_path,
            start_line=start + 1,  # 1‑indexed
            end_line=actual_end,
            content=chunk_text,
        ))
        start += step

    logger.debug("chunk_file: '%s' → %d chunks (chunk_size=%d, overlap=%d)",
                 file_path, len(chunks), chunk_size, overlap)
    return chunks


def chunk_directory(
    files_data: list[dict[str, str]],
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    """
    Chunk all files in `files_data` and return a flat list of chunks.

    Each file record must have at least:
      "file_path": str
      "content":   str

    Args:
      files_data: List of file records.
      chunk_size: Forwarded to `chunk_file`, default 40.
      overlap:    Forwarded to `chunk_file`, default 10.

    Returns:
      Flat list of `Chunk` records, ordered by file and then by line.

    Raises:
      TypeError: If chunk_size or overlap is not int.
      ValueError: If chunk_size/overlap invalid or any record misses required keys.
    """
    _validate_chunk_params(chunk_size, overlap)

    all_chunks: list[Chunk] = []

    for idx, record in enumerate(files_data):
        missing = [k for k in ("file_path", "content") if k not in record]
        if missing:
            raise ValueError(f"files_data[{idx}] missing key(s): {missing}")

        file_chunks = chunk_file(
            record["file_path"],
            record["content"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(file_chunks)

    logger.debug("chunk_directory: %d file(s) → %d total chunks",
                 len(files_data), len(all_chunks))
    return all_chunks
