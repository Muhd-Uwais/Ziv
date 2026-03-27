from pathlib import Path

def load_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None    
    

def load_files_from_directory(root_path: str, extensions: set = {".py"}):
    """
    Recursively load files from directory.

    Args:
        root_path (str): root directory
        extensions (set): allowed extensions (e.g. {".py", ".js"})

    Returns:
        List[dict]: [{"file_path": ..., "content": ...}]
    """

    root = Path(root_path)

    if extensions is None:
        extensions = {".py"}

    files_data = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix not in extensions:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            if content.strip():  # skip empty files
                files_data.append({
                    "file_path": str(file_path),
                    "content": content
                })
        except Exception:
            # skip unreadable files
            continue

    return files_data