from pathlib import Path
from typing import List, Dict


def load_file(file_path: str) -> str | None:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def load_files_from_directory(
        root_path: str,
        extensions: set[str] | None = None
) -> List[Dict[str, str]]:
    """
    Recursively load files from a directory.

    Args:
        root_path: Root directory path.
        extensions: Allowed file extensions (e.g. {".py", ".js"}).

    Returns:
        A list of dictionaries with file path and content.
    """

    if extensions is None:
        extensions = {".py"}

    root = Path(root_path)
    files_data = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix not in extensions:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            if content.strip():  # skip empty files
                files_data.append(
                    {
                        "file_path": str(file_path),
                        "content": content,
                    }
                )
        except (OSError, UnicodeDecodeError):
            # skip unreadable files
            continue

    return files_data
