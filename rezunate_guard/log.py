"""Record problems the user would otherwise never hear about."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rezunate_guard import constants

#: Lines kept. Older ones are dropped, since only a current problem can be acted on.
MAX_LINES = 200

#: How far back to look before repeating ourselves. The hook runs once per tool call.
RECENT_LINES = 20

#: Enough of a problem to act on. A YAML error can run to several lines.
MAX_MESSAGE = 120


def problem(source: Path, message: str) -> None:
    """Add a problem to `~/.rezunate/guard.log`.

    Best effort. Logging must never be the reason a tool call fails.

    Args:
        source: The config file the problem is in.
        message: What is wrong, in the words the user will read.
    """
    text = " ".join(message.split())[:MAX_MESSAGE]
    entry = f"{source}: {text}"

    try:
        path = constants.log_path()
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        if any(line.endswith(entry) for line in lines[-RECENT_LINES:]):
            return

        lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}  {entry}")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text("\n".join(lines[-MAX_LINES:]) + "\n", encoding="utf-8")
    except OSError:
        pass
