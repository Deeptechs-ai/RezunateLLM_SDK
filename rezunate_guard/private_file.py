"""Write a file that only its owner can read."""

from __future__ import annotations

import os
from pathlib import Path


def write(path: Path, data: bytes) -> None:
    """Write `data` to `path`, readable only by this user.

    The mode is set at creation, never after, so the bytes are never briefly readable by
    others. A pid-named temporary file takes the write first, so nobody sees a
    half-written file and two processes cannot collide.

    Args:
        path: The file to write. Its folder is created if missing, mode 0700.
        data: The bytes to write; encode text yourself.

    Raises:
        OSError: If the file could not be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)

    os.replace(temporary, path)
