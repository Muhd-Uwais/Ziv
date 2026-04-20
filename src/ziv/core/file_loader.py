"""
file_loader.py — Recursive source-file discovery for the Ziv pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


_SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
        ".eggs",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "htmlcov",
        ".tox",
        ".idea",
        ".vscode",
    }
)


def _is_venv_dir(path: Path) -> bool:
    """Return True if the directory looks like a Python virtual environment."""
    return (path / "pyvenv.cfg").is_file()


def _should_skip_dir(dir_path: Path) -> bool:
    """Return True if the directory should be pruned from the walk."""
    name = dir_path.name

    if name in _SKIP_DIR_NAMES:
        return True
    if name.endswith(".egg-info"):
        return True
    if _is_venv_dir(dir_path):
        return True
    if "site-packages" in dir_path.parts or "dist-packages" in dir_path.parts:
        return True

    return False


def load_files_from_directory(
    root_path: str | Path,
    extensions: set[str] | None = None,
) -> list[dict[str, str]]:
    """
    Walk a directory and return readable source files.

    Each record has:
      - "file_path": absolute or relative file path as string
      - "content": UTF-8 text content

    Args:
      root_path: Root directory to scan.
      extensions: Allowed file extensions. Defaults to {".py"}.

    Returns:
      A flat list of file records for downstream chunking.

    Raises:
      ValueError: If root_path does not exist or is not a directory.
    """
    root = Path(root_path)

    if not root.exists():
        raise ValueError(f"root_path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"root_path is not a directory: {root}")

    if extensions is None:
        extensions = {".py"}

    normalized_extensions = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in extensions
    }

    files_data: list[dict[str, str]] = []
    skipped_count = 0

    for current_root, dirs, files in os.walk(root):
        current_dir = Path(current_root)

        # Prune in-place so os.walk never descends into ignored directories.
        dirs[:] = [
            directory_name
            for directory_name in dirs
            if not _should_skip_dir(current_dir / directory_name)
        ]

        for filename in files:
            file_path = current_dir / filename

            if file_path.suffix not in normalized_extensions:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("Skipping '%s': %s", file_path, exc)
                skipped_count += 1
                continue

            if not content.strip():
                continue

            files_data.append(
                {
                    "file_path": str(file_path),
                    "content": content,
                }
            )

    if skipped_count:
        logger.debug(
            "load_files_from_directory: skipped %d unreadable file(s) under '%s'",
            skipped_count,
            root,
        )

    logger.debug(
        "load_files_from_directory: loaded %d file(s) from '%s'",
        len(files_data),
        root,
    )

    return files_data
