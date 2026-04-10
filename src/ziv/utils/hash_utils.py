"""hash_utils.py — Deterministic content hashing for the ziv pipeline.

Provides a single stable hash function used across the pipeline wherever a
short, deterministic identifier must be derived from text content — primarily
by the chunking stage to assign stable IDs to each chunk.

SHA-256 is chosen for its collision resistance.  The IDs it produces are used
to detect unchanged chunks on re-indexing; a collision would silently cause
one chunk to overwrite another in the store, so a weak hash (MD5, SHA-1) is
not appropriate here even though the use-case is not cryptographic.
"""

import hashlib


def compute_hash(content: str) -> str:
    """Return the SHA-256 hex digest of *content*.

    **Encoding** — UTF-8 is specified explicitly rather than relying on
    ``str.encode()``'s default, which follows the system locale and differs
    across platforms (notably Windows).  Explicit UTF-8 guarantees the same
    hash for the same logical string on every host.

    **``usedforsecurity=False``** — signals to the underlying OpenSSL backend
    that this hash is used for content addressing, not cryptographic security.
    On FIPS-enabled systems this declaration is required for non-security uses;
    omitting it can cause a ``ValueError`` at runtime on those hosts.

    Output is a 64-character lowercase hex string.  The function is pure and
    deterministic: identical inputs always produce identical outputs.
    """
    return hashlib.sha256(
        content.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
