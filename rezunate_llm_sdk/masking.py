from __future__ import annotations

import json
import re

# Matches a fully-formed placeholder token: [TAG_N] (TAG upper/alnum, N digits).
PLACEHOLDER_RE = re.compile(r"\[[A-Z0-9_]+_\d+\]")


def _normalize_label(label: str) -> str:
    """Make a placeholder tag from a label: ``"email address"`` -> ``"EMAIL_ADDRESS"``."""
    tag = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").upper()
    return tag or "PII"


class MaskVault:
    """A two-way map between PII values and placeholder tokens.

    The same value always gets the same placeholder (so the model sees
    ``[PERSON_1] emailed [PERSON_1]``), and each distinct value gets its own.
    """

    def __init__(self) -> None:
        self._to_placeholder: dict[str, str] = {}  # original -> placeholder
        self._to_original: dict[str, str] = {}  # placeholder -> original
        self._counters: dict[str, int] = {}  # tag -> highest index used

    def assign(self, label: str, original: str) -> str:
        """Return ``original``'s placeholder, minting a new one if unseen."""
        existing = self._to_placeholder.get(original)
        if existing is not None:
            return existing

        tag = _normalize_label(label)
        self._counters[tag] = self._counters.get(tag, 0) + 1
        placeholder = f"[{tag}_{self._counters[tag]}]"
        self._to_placeholder[original] = placeholder
        self._to_original[placeholder] = original
        return placeholder

    def mask_entities(self, text: str, entities: object) -> str:
        """Mask detected spans in ``text`` with placeholders.

        ``entities`` may expose ``start``/``end``/``label`` as attributes (like a
        ``DetectedEntity``) or as dict keys. Spans are replaced right-to-left so
        offsets stay valid; overlapping spans are skipped.
        """
        spans: list[tuple[int, int, str]] = []
        for ent in entities:  # type: ignore[attr-defined]
            if isinstance(ent, dict):
                start, end, label = ent["start"], ent["end"], ent["label"]
            else:
                start, end, label = ent.start, ent.end, ent.label
            spans.append((int(start), int(end), str(label)))

        spans.sort(key=lambda s: s[0], reverse=True)
        result = text
        prev_start: int | None = None
        for start, end, label in spans:
            if start < 0 or end > len(text) or start >= end:
                continue
            # Right-to-left, so an earlier span whose end runs into the previous
            # span's start overlaps it; skip to avoid corrupting the string.
            if prev_start is not None and end > prev_start:
                continue
            original = text[start:end]
            placeholder = self.assign(label, original)
            result = result[:start] + placeholder + result[end:]
            prev_start = start
        return result

    def restore(self, text: str) -> str:
        """Swap every known placeholder in ``text`` back to its original value."""
        if not self._to_original or not text:
            return text
        # Longest placeholder first so no token is a prefix of another.
        for placeholder in sorted(self._to_original, key=len, reverse=True):
            if placeholder in text:
                text = text.replace(placeholder, self._to_original[placeholder])
        return text

    def is_empty(self) -> bool:
        """True if nothing has been masked yet."""
        return not self._to_original

    @property
    def mapping(self) -> dict[str, str]:
        """A read-only copy of the placeholder -> original map (contains PII)."""
        return dict(self._to_original)

    def dumps(self) -> str:
        """Serialize to JSON to persist across turns. Holds real PII — encrypt it."""
        return json.dumps(
            {"map": self._to_original, "counters": self._counters},
            ensure_ascii=False,
        )

    @classmethod
    def loads(cls, data: str) -> MaskVault:
        """Rebuild a vault from ``dumps()`` output to resume a conversation."""
        obj = json.loads(data)
        vault = cls()
        vault._to_original = dict(obj.get("map", {}))
        vault._to_placeholder = {v: k for k, v in vault._to_original.items()}
        vault._counters = {k: int(v) for k, v in obj.get("counters", {}).items()}
        return vault


def unmask(text: str, vault: MaskVault) -> str:
    """Shorthand for ``vault.restore(text)``."""
    return vault.restore(text)


def split_trailing_partial(text: str) -> tuple[str, str]:
    """Hold back a half-typed placeholder from slipping through while streaming.

    When a chunk ends on a lone ``[`` with no closing ``]``, that's probably the
    front of a placeholder whose rest is still on its way — so we wait. You get
    back ``(safe, held)``: go ahead and restore ``safe``, and tack ``held`` onto
    the start of the next chunk.
    """
    idx = text.rfind("[")
    if idx != -1 and "]" not in text[idx:]:
        return text[:idx], text[idx:]
    return text, ""
