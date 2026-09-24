"""Remember what a scan returned, so the same text is never paid for twice.

Keyed by the text itself rather than the file it came from, because an edit changes only
the windows it touches, and most scanned text is a tool's output rather than a file.

Every failure here is answered with a miss. A cache that cannot be read costs a scan; one
that is trusted when it should not be costs a leak.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time

from rezunate_guard import constants

#: How long an answer is reused. The workspace's settings can change without us hearing,
#: so an entry is only ever a day out of date.
TTL_SECONDS = 24 * 60 * 60

#: Rows kept, oldest fetched evicted first. At roughly 120 bytes a row, under 50 MB.
MAX_ENTRIES = 200_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key       TEXT PRIMARY KEY,
    result    TEXT NOT NULL,
    stored_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_stored_at ON entries (stored_at);
"""


def _key(workspace: str, text: str) -> str:
    """Return the row name for one text, scoped to the workspace that scanned it.

    A digest, so no text is ever written to disk. The workspace is part of it because two
    projects have different rules, and one's answers must not stand in for the other's.
    """
    return hashlib.sha256(f"{workspace}\0{text}".encode(errors="replace")).hexdigest()


def _connect() -> sqlite3.Connection | None:
    """Open the cache, creating it owner-only, or return None if it cannot be used."""
    path = constants.cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fresh = not path.exists()
        connection = sqlite3.connect(path, timeout=0.2, isolation_level=None)
        if fresh:
            os.chmod(path, 0o600)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(_SCHEMA)
        return connection
    except (sqlite3.Error, OSError):
        return None


def lookup(workspace: str, texts: list[str]) -> dict[str, dict]:
    """Return the saved result for each text already scanned under this workspace.

    Args:
        workspace: Whose key the texts would be scanned with.
        texts: The texts about to be sent.

    Returns:
        The results still within the retention window, by text. Anything else is a miss,
        including an answer that has gone stale, since a hit is never refreshed by use.
    """
    connection = _connect()
    if connection is None:
        return {}

    keys = {_key(workspace, text): text for text in texts}
    try:
        rows = connection.execute(
            f"SELECT key, result FROM entries "
            f"WHERE stored_at >= ? AND key IN ({','.join('?' * len(keys))})",
            [time.time() - TTL_SECONDS, *keys],
        ).fetchall()
        return {keys[key]: json.loads(result) for key, result in rows}
    except (sqlite3.Error, ValueError):
        return {}
    finally:
        connection.close()


def store(workspace: str, results: dict[str, dict]) -> None:
    """Save what a scan returned, and drop what has expired or overflowed.

    Args:
        workspace: Whose key scanned the texts.
        results: The result for each text, as the scan returned it.
    """
    connection = _connect()
    if connection is None:
        return

    now = time.time()
    try:
        with connection:
            connection.executemany(
                "INSERT OR REPLACE INTO entries (key, result, stored_at) VALUES (?, ?, ?)",
                [
                    (_key(workspace, text), json.dumps(result), now)
                    for text, result in results.items()
                ],
            )
            connection.execute("DELETE FROM entries WHERE stored_at < ?", (now - TTL_SECONDS,))
            connection.execute(
                "DELETE FROM entries WHERE key IN "
                "(SELECT key FROM entries ORDER BY stored_at DESC LIMIT -1 OFFSET ?)",
                (MAX_ENTRIES,),
            )
    except (sqlite3.Error, ValueError, TypeError):
        pass
    finally:
        connection.close()


def summary() -> tuple[int, int]:
    """Return how many answers are held, and the bytes they take."""
    path = constants.cache_path()
    connection = _connect()
    if connection is None:
        return 0, 0
    try:
        entries = connection.execute("SELECT count(*) FROM entries").fetchone()[0]
        return entries, path.stat().st_size
    except (sqlite3.Error, OSError):
        return 0, 0
    finally:
        connection.close()


def clear() -> None:
    """Forget everything, for when an answer is suspect or the disk is wanted back."""
    try:
        constants.cache_path().unlink(missing_ok=True)
    except OSError:
        pass
