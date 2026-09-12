"""Send text to RezunateLLM and get back the PII spans it found."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rezunate_guard import constants

API_PATH = "/api/v1/guardrails/scan-batch"

#: How long one request may take.
TIMEOUT_SECONDS = 180

#: Characters per window.
CHUNK_CHARS = 900

#: Overlap between windows.
OVERLAP_CHARS = 200

#: Most windows per request. The service rejects a larger batch.
MAX_BATCH_TEXTS = 256


class ScanError(Exception):
    """A scan could not be completed. Callers must withhold the content."""


@dataclass(frozen=True)
class Entity:
    """One detected span, with offsets into the full text."""

    start: int
    end: int
    label: str
    score: float


@dataclass(frozen=True)
class ScanResult:
    """What one text scanned to.

    Attributes:
        entities: Detected spans, sorted and disjoint.
        blocked: True when the workspace guardrail blocks rather than redacts.
    """

    entities: tuple[Entity, ...]
    blocked: bool


def chunks(
    text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS
) -> Iterator[tuple[int, str]]:
    """Split `text` into overlapping windows.

    The models truncate without saying so: GLiNER stops at 384 tokens, and anything past
    that comes back with no entities and no error, so a long document reads as clean from
    the middle onward. The overlap means an entity on a boundary is whole in at least one
    window. The service splits what it is given too; doing it twice is worth it because
    the failure is silent.

    Args:
        text: The text to split.
        size: Characters per window.
        overlap: Characters shared by neighbouring windows.

    Yields:
        `(offset, window)`, where `offset` is the window's start in `text`.

    Raises:
        ValueError: If `size` is not larger than `overlap`, which would never advance.
    """
    if size <= overlap:
        raise ValueError("chunk size must exceed the overlap, or this never advances")

    start = 0
    while True:
        end = min(start + size, len(text))
        yield start, text[start:end]
        if end >= len(text):
            return
        start = end - overlap


def drop_overlaps(entities: Iterable[Entity]) -> tuple[Entity, ...]:
    """Reduce raw detections to spans that are sorted and disjoint.

    Overlaps come from two places: neighbouring windows report anything in their overlap
    twice, and the models can tag the same words under two labels. Masking handles
    neither, since overlapping spans shift every later offset and corrupt the text.

    One span survives a clash: the longest, because it hides the most, with ties going
    to the higher score. The sort puts them in that order, so one pass is enough.

    Args:
        entities: Detected spans, in any order.

    Returns:
        Spans sorted by start, with no two overlapping.
    """
    kept: list[Entity] = []
    for entity in sorted(entities, key=lambda e: (e.start, -e.end, -e.score)):
        if not kept or entity.start >= kept[-1].end:
            kept.append(entity)
    return tuple(kept)


def stored_key() -> str:
    """Return the key saved by `rezunate-guard login`, or "" if there is none."""
    try:
        return constants.credentials_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _credentials() -> tuple[str, str]:
    """Return the API key and base URL to scan with.

    The environment beats the stored key, so a project can point at another workspace
    without rewriting the file.

    Returns:
        `(api_key, base_url)`.

    Raises:
        ScanError: If no key is set anywhere.
    """
    api_key = os.environ.get(constants.API_KEY_ENV, "").strip() or stored_key()
    base_url = os.environ.get(constants.BASE_URL_ENV, constants.DEFAULT_BASE_URL).rstrip("/")
    if not api_key:
        raise ScanError(f"no API key; run `rezunate-guard login` or set {constants.API_KEY_ENV}")
    return api_key, base_url


def send_batch(texts: list[str]) -> list[dict]:
    """Send one request covering every text in the batch.

    Uses urllib rather than requests to keep the package free of dependencies. A cache
    will wrap this call later.

    Args:
        texts: Texts to scan.

    Returns:
        One payload per text, in the same order.

    Raises:
        ScanError: On any network, HTTP or decoding failure, or if the reply does not
            hold one result per text.
    """
    api_key, base_url = _credentials()
    request = Request(
        f"{base_url}{API_PATH}",
        data=json.dumps({"texts": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ScanError(f"scan failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ScanError(f"could not reach {base_url}: {exc.reason}") from exc
    except (ValueError, OSError) as exc:
        raise ScanError(f"scan failed: {exc}") from exc

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != len(texts):
        raise ScanError("scan response did not carry one result per text")
    return results


def _parse_entities(payload: dict, offset: int) -> list[Entity]:
    """Read entities out of one payload.

    Args:
        payload: One text's result.
        offset: Where that text's window starts in the full text.

    Returns:
        Entities with offsets shifted into the full text.

    Raises:
        ScanError: If the payload has no entity list or an entity is malformed.
    """
    raw = payload.get("entities")
    if not isinstance(raw, list):
        raise ScanError("response had no entity list")

    try:
        return [
            Entity(
                start=int(item["start"]) + offset,
                end=int(item["end"]) + offset,
                label=str(item["label"]),
                score=float(item.get("score", 0.0)),
            )
            for item in raw
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ScanError(f"malformed entity in response: {exc}") from exc


def scan_many(texts: Sequence[str]) -> tuple[ScanResult, ...]:
    """Scan several texts together, one result per input.

    Every window of every text goes into one request, so a grep with a hundred matched
    lines costs one call instead of a hundred. Repeated strings are sent once and still
    mask alike, because a placeholder comes from the value.

    Args:
        texts: Texts to scan. Blank ones are skipped but still get a result.

    Returns:
        One `ScanResult` per input text, in order, with offsets into that text.

    Raises:
        ScanError: If a reply is malformed or the wrong length.
    """
    # Windows are cut here as well as in the service, because truncation is silent.
    windows: list[tuple[int, int, str]] = []
    for index, text in enumerate(texts):
        if not text.strip():
            continue
        windows.extend((index, offset, window) for offset, window in chunks(text))

    if not windows:
        return tuple(ScanResult(entities=(), blocked=False) for _ in texts)

    unique: dict[str, int] = {}
    for _, _, window in windows:
        unique.setdefault(window, len(unique))

    ordered = list(unique)
    payloads: list[dict] = []
    for start in range(0, len(ordered), MAX_BATCH_TEXTS):
        group = ordered[start : start + MAX_BATCH_TEXTS]
        answered = send_batch(group)
        if not isinstance(answered, list) or len(answered) != len(group):
            raise ScanError("scan response did not carry one result per text")
        payloads.extend(answered)

    found: list[list[Entity]] = [[] for _ in texts]
    blocked = [False] * len(texts)
    for index, offset, window in windows:
        payload = payloads[unique[window]]
        if not isinstance(payload, dict):
            raise ScanError("response was not an object")
        blocked[index] = blocked[index] or bool(payload.get("blocked"))
        found[index].extend(_parse_entities(payload, offset))

    return tuple(
        ScanResult(entities=drop_overlaps(entities), blocked=was_blocked)
        for entities, was_blocked in zip(found, blocked, strict=True)
    )


def scan(text: str) -> ScanResult:
    """Scan one text.

    Args:
        text: The text to scan.

    Returns:
        The result for that text.

    Raises:
        ScanError: As `scan_many`.
    """
    return scan_many([text])[0]
