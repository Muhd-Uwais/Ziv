import uuid
from typing import List, Dict


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = 40,
    overlap: int = 10,
) -> List[Dict[str, str | int]]:
    """
    Split file content into overlapping chunks.

    Args:
        file_path: Path of the file.
        content: Full file content.
        chunk_size: Number of lines per chunk.
        overlap: Overlapping lines between chunks.

    Returns:
        List of chunk metadata dictionaries.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    lines = content.splitlines()

    # Edge case: empty file
    if not lines:
        return []

    chunks = []
    step = chunk_size - overlap
    total_lines = len(lines)

    # Edge case: small files
    if total_lines <= chunk_size:
        chunk_text = "\n".join(lines)
        return [
            {
                "id": str(uuid.uuid4()),
                "file_path": file_path,
                "start_line": 1,
                "end_line": total_lines,
                "content": chunk_text,
            }
        ]

    # Sliding window
    start = 0
    while start < total_lines:
        end = start + chunk_size
        chunk_lines = lines[start:end]

        # Edge case: empty chunk (shouldn't happen but safe)
        if not chunk_lines:
            break

        chunk_text = "\n".join(chunk_lines)

        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "file_path": file_path,
                "start_line": start + 1,
                "end_line": min(end, total_lines),
                "content": chunk_text,
            }
        )

        # Move window
        start += step

    return chunks


def chunks_directory(
    files_data: list[dict],
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[dict]:

    files_chunk = []
    
    for data in files_data:
        file_path = data['file_path']
        content = data['content']

        files_chunk += chunk_file(file_path, content, chunk_size, overlap)

    return files_chunk
