"""
file_loader.py — Recursive source-file discovery for the ziv pipeline.

Walks a directory tree, prunes paths irrelevant to user source code
(virtual environments, caches, build artefacts, dependency trees), and
reads surviving files into the ``{"file_path", "content"}`` records
expected by the chunking stage.

Pruning is the performance-critical property here: ``dirs`` is modified
in-place inside ``os.walk`` so that entire subtrees are never descended
into, rather than being visited and discarded after the fact.  For
repositories that vendor large dependency trees (e.g., a project with a
committed ``node_modules`` or a non-standard ``.venv`` name) the
difference is traversal of tens of thousands of files versus a handful.
"""

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skip-list constants
# ---------------------------------------------------------------------------


# Exact directory names (not full paths) that are never user source.
# Matching on the final path component only means a project directory
# literally named "build" is also excluded — that trade-off is intentional
# and acceptable for the CLI's target use-case (indexing active source trees).
_SKIP_DIR_NAMES: frozenset[str] = frozenset({
    # Version-control metadata
    ".git", ".hg", ".svn",
    # Python byte-code and tool caches
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    # Build and distribution artefacts
    "build",
    "dist",
    ".eggs",
    # Common virtual-environment root names (custom names caught by _is_venv_dir)
    ".venv",
    "venv",
    "env",
    # JavaScript / Node dependency tree
    "node_modules",
    # Coverage reports and multi-environment test runners
    "htmlcov",
    ".tox",
    # IDE project files
    ".idea",
    ".vscode",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_venv_dir(path: Path) -> bool:
    """Return True if *path* is the root of a Python virtual environment.

    ``pyvenv.cfg`` is the canonical venv marker (PEP 405).  Checking for it
    catches virtual environments with non-standard names (e.g., ``my-env``,
    ``.project-venv``) that would slip past the static ``_SKIP_DIR_NAMES``
    lookup.
    """
    return (path / "pyvenv.cfg").is_file()


def _should_skip_dir(dir_path: Path) -> bool:
    """Return True if *dir_path* should be excluded from the file walk.

    Checks are ordered cheapest-first to avoid filesystem I/O when a
    cheap name lookup is sufficient:

    1. **Name in static skip-list** — O(1) frozenset lookup.  Covers the
       vast majority of cases without any filesystem access.
    2. **Name ends with** ``.egg-info`` — installed-package metadata dirs
       follow ``<pkg>-<ver>.egg-info`` naming and cannot be listed
       exhaustively in the static set.
    3. **``pyvenv.cfg`` is present** — filesystem probe (one stat call) that
       handles custom-named virtual environments.
    4. **Path component check for site-packages / dist-packages** — guards
       against non-standard venv layouts.  Deliberately checks *components*
       (``Path.parts``) rather than a raw substring to avoid a false positive
       when a legitimate parent directory contains those words in its name
       (e.g., ``/home/user/my-site-packages-tool/src``).
    """
    name = dir_path.name

    # 1. Fast static lookup — no I/O
    if name in _SKIP_DIR_NAMES:
        return True

    # 2. Egg-info dirs can't be enumerated statically; match by suffix
    if name.endswith(".egg-info"):
        return True

    # 3. Actual venv root — one stat call, only reached if name is unknown
    if _is_venv_dir(dir_path):
        return True

    # 4. Deep inside a venv or system-Python layout
    if "site-packages" in dir_path.parts or "dist-packages" in dir_path.parts:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_files_from_directory(
    root_path: str | Path,
    extensions: set[str] | None = None,
) -> list[dict[str, str]]:
    """Walk *root_path* and return readable source files as a flat record list.

    Each record is ``{"file_path": <str>, "content": <str>}`` — the contract
    expected by the chunking stage.

    **Pruning** — ``dirs`` is modified in-place inside ``os.walk`` so whole
    subtrees are skipped without descent.  A list comprehension is used (not
    a set comprehension) to preserve the deterministic traversal order that
    ``os.walk`` guarantees; set iteration order is undefined.

    **Unreadable files** — ``OSError`` (permission denied, broken symlink)
    and ``UnicodeDecodeError`` (binary or non-UTF-8 content) are caught,
    logged at DEBUG, and skipped.  The walk continues rather than aborting
    on the first problematic file; a final DEBUG log reports the total count.

    **Empty files** — whitespace-only files are excluded before appending;
    they produce chunks with no indexable signal.

    **Extension normalisation** — a leading dot is inserted if absent, so
    contributors passing ``{"py"}`` and ``{".py"}`` get identical behaviour
    rather than a silent no-match.

    Raises:
        ValueError: If *root_path* does not exist or is not a directory.
            Raised eagerly so the CLI can surface a clear error rather than
            returning an empty list that looks like a successful empty scan.
    """
    root = Path(root_path)

    # Validate upfront — os.walk silently yields nothing for missing paths,
    # which would look like a successful empty scan to callers.
    if not root.exists():
        raise ValueError(f"root_path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"root_path is not a directory: {root}")

    if extensions is None:
        extensions = {".py"}

    # Normalise extensions: contributors may omit the leading dot.
    extensions = {
        ext if ext.startswith(".") else f".{ext}" for ext in extensions
    }

    files_data: list[dict[str, str]] = []
    skipped_count = 0

    for root_str, dirs, files in os.walk(root):
        current_dir = Path(root_str)

        # Prune in-place — os.walk will not descend into removed entries.
        # List comprehension preserves original traversal order; a set
        # comprehension here would make directory visit order non-deterministic.
        dirs[:] = [
            d for d in dirs
            if not _should_skip_dir(current_dir / d)
        ]

        for filename in files:
            file_path = current_dir / filename

            if file_path.suffix not in extensions:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("Skipping '%s': %s", file_path, exc)
                skipped_count += 1
                continue

            # Exclude whitespace-only files — no indexable content
            if not content.strip():
                continue

            files_data.append({
                "file_path": str(file_path),
                "content": content,
            })

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
