"""Hash helpers for deterministic content fingerprints in Ziv."""

from __future__ import annotations

import hashlib


def compute_hash(content: str) -> str:
    """Return a deterministic SHA-256 hex digest for text content."""
    return hashlib.sha256(
        content.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()