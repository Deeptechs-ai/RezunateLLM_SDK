"""Keep redacted copies for files the model cannot be shown directly.

A PDF's pages never appear in the tool result, so they cannot be rewritten there. What
can be done is to point the read at a redacted copy instead. This is where that copy
lives.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from rezunate_guard import constants

#: Told to the model at the top of every redacted copy, so it never reports extracted text as
#: the file itself.
HEADER = (
    "[rezunate-guard] This is the text of {name}, with personal data replaced by\n"
    "placeholders. It is not the original file: layout, images and anything the\n"
    "extractor could not read are missing.\n"
    "\n"
)

#: Enough of the digest to name a file without collisions worth worrying about.
NAME_DIGITS = 16


def _content_fingerprint(original: Path, text: str) -> str:
    """Name a redacted copy after what is in it, so the same file reuses the same copy."""
    material = f"{original}\0{text}".encode("utf-8", errors="replace")
    return hashlib.sha256(material).hexdigest()[:NAME_DIGITS]


def write(original: Path | str, redacted: str) -> Path:
    """Write a redacted redacted copy and return the path to read instead.

    The write goes through a temporary file so a reader can never catch a half-written
    redacted copy, and the mode is set at creation rather than after, so the text is never
    briefly readable by other users.

    Args:
        original: The file being stood in for. Named in the header.
        redacted: Its text, already redacted.

    Returns:
        The redacted copy's path.

    Raises:
        OSError: If it could not be written.
    """
    original = Path(original)
    body = HEADER.format(name=original.name) + redacted

    directory = constants.redacted_copies_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f"{original.stem}-{_content_fingerprint(original, redacted)}.txt"

    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.replace(temporary, path)
    return path
