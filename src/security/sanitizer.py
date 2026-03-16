"""
Input sanitisation utilities for slide content before it reaches LLM prompts.

Slide content is untrusted user input.  Before injecting it into any LLM
prompt, all text must pass through sanitize_slide_content().

What this guards against:
  - Prompt injection: instructions embedded in slides ("ignore previous…")
  - Control characters that can confuse tokenisers or break JSON responses
  - Excessively long inputs that inflate cost and risk jailbreak padding
  - Null bytes and path traversal characters in filenames
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Prompt injection patterns ─────────────────────────────────────────────────
# Heuristic patterns only — not a complete defence, but catches obvious attacks.
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions?"
    r"|forget\s+(all\s+)?instructions?"
    r"|you\s+are\s+now\s+a"
    r"|system\s*:\s*you"
    r"|<\s*/?system\s*>"
    r"|###\s*(system|instruction|prompt))",
    re.IGNORECASE,
)

# ── Control characters (except \t \n \r which are legitimate whitespace) ──────
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# ── Filename safety ───────────────────────────────────────────────────────────
_UNSAFE_FILENAME = re.compile(r"[^\w.\-_ ]")  # allow word chars, dot, dash, underscore, space
_PATH_TRAVERSAL = re.compile(r"\.\./|\.\.\\|^/|^\\")


def sanitize_slide_content(text: str, max_len: int = 5000) -> str:
    """
    Sanitise raw slide text before including it in an LLM prompt.

    Steps:
      1. Remove C0/C1 control characters.
      2. Detect and redact prompt-injection patterns (with a warning log).
      3. Truncate to max_len characters.

    Does NOT raise — always returns a safe string.
    """
    if not text:
        return ""

    # Step 1: strip control chars (preserve \t \n \r)
    text = _CONTROL_CHARS.sub("", text)

    # Step 2: redact injection attempts
    if _INJECTION_PATTERNS.search(text):
        logger.warning(
            "Potential prompt injection detected in slide content (first 100 chars): %.100s",
            text,
        )
        text = _INJECTION_PATTERNS.sub("[REDACTED]", text)

    # Step 3: truncate
    return text[:max_len]


def sanitize_filename(filename: str) -> str:
    """
    Return a safe version of an upload filename.

    - Strips path components (directory traversal prevention).
    - Replaces unsafe characters with underscores.
    - Limits length.
    """
    # Take only the basename component
    basename = re.split(r"[/\\]", filename)[-1]

    # Replace unsafe characters
    safe = _UNSAFE_FILENAME.sub("_", basename)

    # Truncate
    return safe[:255]


def validate_filename(filename: str) -> None:
    """
    Raise ValueError if the filename looks like a path traversal attempt
    or contains null bytes.
    """
    if "\x00" in filename:
        raise ValueError("Filename contains null byte")
    if _PATH_TRAVERSAL.search(filename):
        raise ValueError(f"Filename contains path traversal sequence: {filename!r}")


def check_pptx_magic_bytes(data: bytes) -> bool:
    """
    Return True if data starts with PPTX magic bytes (PK ZIP header).

    PPTX files are ZIP archives; all start with b'PK\x03\x04'.
    This provides a minimal content-type check independent of the filename.
    """
    return data[:4] == b"PK\x03\x04"
