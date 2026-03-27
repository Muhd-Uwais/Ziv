import uuid


def chunk_file(file_path: str, content: str, chunk_size: int = 40, overlap: int = 10):
    """
    Split file content into overlapping chunks.

    Args:
        file_path (str): Path of the file.
        content (str): Full file content
        chunk_size (int): Number of lines per chunk
        overlap (int): Overlapping lines between chunks

    Returns:
        List[dict]: List of chunk metadata    
    """

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
        chunks.append({
            "id": str(uuid.uuid4()),
            "file_path": file_path,
            "start_line": 1,
            "end_line": total_lines,
            "content": chunk_text
        })
        return chunks
    
    # Sliding window
    start = 0
    while start < total_lines:
        end = start + chunk_size
        chunk_lines = lines[start:end]

        # Edge case: empty chunk (shouldn't happend but safe)
        if not chunk_lines:
            break

        chunk_text = "\n".join(chunk_lines)

        chunks.append({
            "id": str(uuid.uuid4()),
            "file_path": file_path,
            "start_line": start + 1,
            "end_line": min(end, total_lines),
            "content": chunk_text
        })

        # Move window
        start += step

    return chunks    


def chunks_directory(files_data, chunk_size: int = 40, overlap: int = 10):

    files_chunk = []
    i = 0
    for data in files_data:
        file_path = data['file_path']
        content = data['content']

        files_chunk += chunk_file(file_path, content, chunk_size, overlap)

    return files_chunk