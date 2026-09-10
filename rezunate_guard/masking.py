"""Replace detected PII with placeholders, so the model reads the text but not the
values in it."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Iterable

from rezunate_guard import constants, private_file
from rezunate_guard.scanner import Entity


class MaskingError(Exception):
    """Spans could not be masked safely. Callers must withhold the content."""


#: A complete placeholder: [TAG_abcd].
PLACEHOLDER_RE = re.compile(r"\[[A-Z0-9_]+_[0-9a-f]+\]")

#: Hex digits per placeholder. Four gives 65,536 tokens per label, so a clash needs a
#: few hundred people in one document. A clash merges two placeholders but leaks nothing.
SUFFIX_DIGITS = 4

_NON_TAG_CHARS = re.compile(r"[^A-Za-z0-9]+")


def _tag(label: str) -> str:
    """Turn a label into a tag.

    Args:
        label: A model label, such as "email address".

    Returns:
        The tag to put in the placeholder, such as "EMAIL_ADDRESS".
    """
    return _NON_TAG_CHARS.sub("_", label.strip()).strip("_").upper() or "PII"


def placeholder_key() -> bytes:
    """Return this machine's placeholder key, creating it on first use.

    The key only stops someone guessing which values a redacted document hides, but it
    must stay stable or the same person gets a different placeholder in every file.

    Tool calls run at the same time, so two processes can both try to create it. Each
    writes its own temporary file and the last rename wins. The key is read back
    afterwards so every process returns the winner's key, not the one it made.

    Returns:
        32 random bytes, from `~/.rezunate/placeholder.key` (mode 0600).
    """
    path = constants.placeholder_key_path()
    try:
        if existing := path.read_bytes():
            return existing
    except OSError:
        pass

    private_file.write(path, secrets.token_bytes(32))
    return path.read_bytes()


def placeholder(label: str, value: str, key: bytes) -> str:
    """Build the placeholder that stands in for one value.

    Args:
        label: The model's label for the value.
        value: The detected text.
        key: The placeholder key.

    Returns:
        A placeholder such as `[PERSON_7f3a]`.
    """
    digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"[{_tag(label)}_{digest[:SUFFIX_DIGITS]}]"


def mask(text: str, entities: Iterable[Entity], key: bytes | None = None) -> str:
    """Replace every entity span in `text` with its placeholder.

    Works right to left, so replacing one span never moves the spans still to come.
    Spans outside the text are skipped.

    Args:
        text: The text to mask.
        entities: Spans to replace. Must be disjoint; `scanner.drop_overlaps` makes
            them so.
        key: The placeholder key. Read from disk when omitted.

    Returns:
        The text with each span replaced.

    Raises:
        MaskingError: If two spans overlap. Masking them would shift every later offset
            and corrupt the text instead of protecting it.
    """
    if key is None:
        key = placeholder_key()

    result = text
    previous_start = len(text)
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        start, end = entity.start, entity.end
        if start < 0 or end > len(text) or start >= end:
            continue
        if end > previous_start:
            raise MaskingError(
                f"overlapping spans reached the masker at {start}:{end}; "
                "scanner.drop_overlaps should have resolved this"
            )
        result = result[:start] + placeholder(entity.label, text[start:end], key) + result[end:]
        previous_start = start
    return result
