"""File handling utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hash of a file (for caching and deduplication)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str, max_len: int = 80) -> str:
    """Sanitise a filename for safe storage."""
    import re
    safe = re.sub(r"[^\w\s\-.]", "_", name)
    safe = re.sub(r"\s+", "_", safe)
    return safe[:max_len]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and all parents if they don't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
