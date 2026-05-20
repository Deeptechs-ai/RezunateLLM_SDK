"""SSE (Server-Sent Events) line parser.

Takes an iterator of decoded text lines and yields
:class:`SSEEvent` objects, one per dispatched event.
"""

from collections.abc import Iterator
from typing import NamedTuple


class SSEEvent(NamedTuple):
    """A single dispatched Server-Sent Event frame."""

    event: str
    data: str


def _flush(event: str, data_buf: list[str]) -> SSEEvent | None:
    if not data_buf:
        return None
    return SSEEvent(event=event or "message", data="\n".join(data_buf))


def parse_sse_lines(lines: Iterator[str]) -> Iterator[SSEEvent]:
    """Parse SSE frames from an iterator of text lines.

    Yields one :class:`SSEEvent` per dispatched event. Comments (lines
    starting with ``:``) and unknown fields are ignored, matching the
    WHATWG spec.
    """
    event = ""
    data_buf: list[str] = []

    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if (out := _flush(event, data_buf)) is not None:
                yield out
            event = ""
            data_buf = []
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""
        if field == "event":
            event = value
        elif field == "data":
            data_buf.append(value)
        # id, retry, and unknown fields ignored

    if (out := _flush(event, data_buf)) is not None:
        yield out
