"""The hook: the last point where PII can be stopped before the model sees it."""

from __future__ import annotations

import codecs
import glob
import json
import os
import re
import sys
from typing import Any

from rezunate_guard.config import should_scan
from rezunate_guard.masking import MaskingError, mask, placeholder_key
from rezunate_guard.scanner import ScanError, scan_many

#: Sent in place of content we could not redact. It must not claim the read was
#: prevented, since on the PostToolUse path it already happened.
WITHHELD_NOTICE = (
    "[rezunate-guard] Protected content was withheld here: {reason}.\n"
    "None of it is available to you; do not guess at what it contained. "
    "Ask the user to run `rezunate-guard status` for details."
)

#: Keys under which a tool names the file it was pointed at.
PATH_KEYS = ("file_path", "notebook_path", "path")

#: Keys whose value is a path. Redacting one breaks the model's handle on the file, and
#: a path is not where prose hides.
PATH_VALUED_KEYS = frozenset({"filePath", "file_path", "outputDir"})

#: Enough of a file to spot a binary header, cheap enough to read on every check.
SNIFF_BYTES = 8192

#: What a discriminator looks like: one short lowercase word.
_TAG_VALUE = re.compile(r"[a-z][a-z0-9_-]{0,31}")

_COMMAND_TOKEN = re.compile(r"[^\s;|&()<>\"'`]+")
_LOOKS_LIKE_FILE = re.compile(r"\.\w{1,8}$")


class Blocked(Exception):
    """The workspace guardrail blocks this content rather than redacting it."""


def redact_all(texts: list[str]) -> dict[str, str]:
    """Redact many strings in one request.

    Scanning together is the point: a tool result is many short strings, and one request
    each turned a single file read into dozens of round trips. Repeats are scanned once
    and still mask alike, because a placeholder comes from the value.

    Args:
        texts: Strings to redact.

    Returns:
        Each unique string mapped to its redacted form.

    Raises:
        ScanError: If the scan fails.
        Blocked: If the workspace guardrail blocks any of it.
    """
    unique = list(dict.fromkeys(texts))
    results = scan_many(unique)
    if any(result.blocked for result in results):
        raise Blocked("the workspace guardrail is set to block this content")

    # One read of the key for the whole result rather than one per string.
    key = placeholder_key()
    return {
        text: mask(text, result.entities, key) if result.entities else text
        for text, result in zip(unique, results, strict=True)
    }


def redact_response(response: Any) -> Any:
    """Copy a tool response with every string in it redacted.

    Runs the same walker twice, so what counts as content is written once: one pass
    collects the strings, they are scanned together, the next puts the results back.

    Args:
        response: The tool response, of any shape.

    Returns:
        A copy with each string redacted.

    Raises:
        ScanError: If the scan fails.
        Blocked: If the workspace guardrail blocks the content.
    """
    pending = _strings_in(response)
    if not pending:
        return response

    redacted = redact_all(pending)
    return _rewrite_strings(response, lambda text: redacted[text])


def _strings_in(response: Any) -> list[str]:
    """Return every string the redactor treats as content, in the order it finds them."""
    found: list[str] = []

    def note(text: str) -> str:
        found.append(text)
        return text

    _rewrite_strings(response, note)
    return found


def _paths_in_command(command: str) -> list[str]:
    """Find the paths a shell command refers to.

    A heuristic, since a path can come from a variable we cannot follow. It covers
    `cat clients/acme.md`, which is how protected files get read by something other
    than Read.

    Args:
        command: The shell command.

    Returns:
        Paths as written, in order. Possibly empty.
    """
    found: list[str] = []
    for raw in _COMMAND_TOKEN.findall(command):
        token = raw.strip("'\"`,:")
        # Split on "=" first, or the "-" test below discards a flag before its path is
        # ever looked at.
        if "=" in token and not token.startswith("/"):
            token = token.split("=", 1)[-1]
        if not token or token.startswith("-") or "://" in token:
            continue
        if "/" not in token and not _LOOKS_LIKE_FILE.search(token):
            continue

        found.append(token)
    return found


def _tool_path(payload: dict[str, Any]) -> str | None:
    """Return the path the tool was pointed at, for replies that need one."""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    for key in PATH_KEYS:
        path = tool_input.get(key)
        if isinstance(path, str) and path:
            return path
    return None


def _implicated_paths(payload: dict[str, Any]) -> list[str]:
    """Find every path the call may have touched.

    Taken from the file the tool was given and the paths named in the command it ran, so
    `cat`, `grep` and an MCP reader go through the same check as `Read`.

    Args:
        payload: The hook payload.

    Returns:
        Absolute paths, resolved against the call's `cwd`, with globs expanded.
    """
    paths: list[str] = []

    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        paths.extend(
            value for key in PATH_KEYS if isinstance(value := tool_input.get(key), str) and value
        )
        command = tool_input.get("command")
        if isinstance(command, str):
            paths.extend(_paths_in_command(command))

    response = payload.get("tool_response")
    if isinstance(response, dict):
        file = response.get("file")
        if isinstance(file, dict) and isinstance(file.get("filePath"), str):
            paths.append(file["filePath"])

    cwd = payload.get("cwd")
    base = cwd if isinstance(cwd, str) else os.getcwd()
    resolved = (os.path.join(base, os.path.expanduser(path)) for path in paths)
    return [candidate for path in resolved for candidate in _expand_glob(path)]


def _expand_glob(path: str) -> list[str]:
    """Return a glob along with the files it matches.

    Both halves are needed. The pattern is what a config entry matches as a literal
    string; the matches are what PreToolUse opens, since `isfile("clients/*.pdf")` is
    False.

    Args:
        path: An absolute path, which may hold glob characters.

    Returns:
        Just the path if it holds none, otherwise the path and its matches.
    """
    if not any(character in path for character in "*?["):
        return [path]
    return [path, *sorted(glob.glob(path))]


def _reaches_the_model_as_text(path: str) -> bool:
    """Check whether reading this file gives text a PostToolUse hook could rewrite.

    Decided from the bytes, not the extension, so PDFs, images, archives and formats
    nobody has thought of are covered without a list to maintain.

    Args:
        path: The file to inspect.

    Returns:
        True if the first few KB read as text. A file we cannot open counts as False,
        since a protected file we cannot inspect should be stopped.
    """
    try:
        with open(path, "rb") as handle:
            sample = handle.read(SNIFF_BYTES)
    except OSError:
        return False

    if b"\x00" in sample:
        return False

    try:
        # An incremental decoder forgives a character cut in half by the sample boundary,
        # which a plain decode would call binary.
        codecs.getincrementaldecoder("utf-8")().decode(sample)
    except UnicodeDecodeError:
        return False
    return True


def _deny(reason: str) -> dict[str, Any]:
    """Build the PreToolUse reply that refuses a call outright."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _before_tool(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Refuse calls whose content could not be redacted afterwards.

    Returns:
        A deny reply, or None to allow the call.
    """
    for path in _implicated_paths(payload):
        if not os.path.isfile(path):
            continue  # a folder, or a path that does not exist yet
        if not should_scan(path) or _reaches_the_model_as_text(path):
            continue
        return _deny(
            f"{os.path.basename(path)} is protected, and its contents cannot be "
            "redacted after reading: they reach the model outside the tool result, "
            "where rezunate-guard cannot rewrite them. Read the file some other way, "
            "or take it out of the `scan:` list in .rezunate-guard.yaml."
        )
    return None


def _rewrite_strings(value: Any, transform, field: str | None = None) -> Any:
    """Copy a value, running `transform` over every string in it.

    Walking the whole response rather than reaching for a known field is what lets one
    function serve Read's nested `file.content`, Bash's flat `stdout`, a bare string and
    whatever an MCP server invents, without knowing any of their schemas.

    Copying rather than rebuilding matters: Claude Code checks the reply against the
    tool's schema and throws away a mismatch, sending the original unredacted output.

    Args:
        value: Any part of a tool response.
        transform: Called with each content string, returning its replacement.
        field: The key this value was found under. Tells structure from content.

    Returns:
        A copy with every content string replaced.
    """
    if isinstance(value, str):
        if not value or field in PATH_VALUED_KEYS:
            return value
        # A "type" holds the tag the schema checks, at any depth, and rewriting one gets
        # the whole reply rejected — a worse leak than the tag could be. Only when it
        # looks like a tag, since "type" is also an ordinary word.
        if field == "type" and _TAG_VALUE.fullmatch(value):
            return value
        return transform(value)

    if isinstance(value, dict):
        return {name: _rewrite_strings(item, transform, name) for name, item in value.items()}

    if isinstance(value, list):
        # A list does not name its items, so they keep their parent's field.
        return [_rewrite_strings(item, transform, field) for item in value]

    return value


def _reply(updated: Any) -> dict[str, Any]:
    """Build a PostToolUse reply that replaces the tool output."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": updated,
        }
    }


def _is_image(payload: dict[str, Any]) -> bool:
    """Return True if the tool returned an image."""
    response = payload.get("tool_response")
    if not isinstance(response, dict):
        return False
    return response.get("type") == "image" or response.get("isImage") is True


def _content_is_elsewhere(response: Any) -> bool:
    """Return True for a file result holding no text to rewrite.

    A PDF is the usual case: its pages are counted, but its text never appears.
    """
    if not isinstance(response, dict):
        return False
    file = response.get("file")
    return isinstance(file, dict) and not isinstance(file.get("content"), str)


def _replace_content(payload: dict[str, Any], text: str) -> dict[str, Any]:
    """Build a Read reply carrying `text` in place of the file's content.

    The response is copied rather than rebuilt, so its shape cannot drift.
    """
    response = payload["tool_response"]
    return _reply({**response, "file": {**response["file"], "content": text}})


def _as_text_response(path: str, text: str) -> dict[str, Any]:
    """Build a text-shaped Read result from scratch.

    For replies with no content field to overwrite. Claude Code uses this same shape when
    it strips an image from a transcript, so Read's schema accepts it.
    """
    return {
        "type": "text",
        "file": {
            "filePath": path,
            "content": text,
            "numLines": 1,
            "startLine": 1,
            "totalLines": 1,
        },
    }


def _holds_content(response: Any) -> bool:
    """Return True if a response carries any string we would treat as content.

    Walks with the redactor, so the two can never disagree about what counts.
    """
    return any(text.strip() for text in _strings_in(response))


def _withhold(payload: dict[str, Any], reason: str) -> dict[str, Any] | None:
    """Drop the content, keeping a shape the tool accepts.

    Read gets its own branch, since a notice in `file.content` reads naturally. Other
    tools get the notice in their first non-empty string, with the rest emptied. Shape
    matters as much here as in redaction: a rejected reply sends the raw output through.

    Args:
        payload: The hook payload.
        reason: Why the content was withheld. Shown to the model.

    Returns:
        A reply carrying the notice, or None only when there was no content to withhold.
        `main` prints nothing for None, so the two must never be confused.
    """
    notice = WITHHELD_NOTICE.format(reason=reason)
    response = payload.get("tool_response")
    path = _tool_path(payload)

    if (
        isinstance(response, dict)
        and response.get("type") == "text"
        and isinstance(response.get("file"), dict)
    ):
        return _replace_content(payload, notice)

    if _is_image(payload) and path is not None:
        # Blanking the base64 in place would leave an image-shaped result holding
        # nothing, so send a text response instead.
        return _reply(_as_text_response(path, notice))

    # Asked before blanking, not discovered afterwards. Falling through on a response
    # that did hold content would leave the original standing.
    if not _holds_content(response):
        return None

    placed = False

    def blank(text: str) -> str:
        nonlocal placed
        if not placed and text.strip():
            placed = True
            return notice
        return ""

    return _reply(_rewrite_strings(response, blank))


def respond(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Decide what to send back for one hook event.

    PostToolUse rewrites what a tool returned, which covers anything arriving as text.
    PreToolUse refuses the call, the only answer for a file whose contents never appear
    in the result: a PDF arrives as page images in a message we are never shown.

    The tool's name never matters, only whether the call touched a protected path.

    Args:
        payload: The hook payload from stdin.

    Returns:
        The reply to print, or None to leave the tool output alone.

    Raises:
        ScanError: If the scan fails.
        Blocked: If the workspace guardrail blocks the content.
    """
    event = payload.get("hook_event_name")
    if event == "PreToolUse":
        return _before_tool(payload)
    if event != "PostToolUse":
        return None

    response = payload.get("tool_response")
    if response is None:
        return None

    if not any(should_scan(path) for path in _implicated_paths(payload)):
        return None

    if _is_image(payload):
        return _withhold(payload, "images cannot be redacted")

    if _content_is_elsewhere(response):
        # PreToolUse should have refused this. If it is not installed, withholding is all
        # that is left, since the content reaches the model by a route we never see.
        return _withhold(payload, "this result holds content we cannot rewrite")

    return _reply(redact_response(response))


def _reason_for(exc: BaseException) -> str:
    """Turn an exception into a reason the user can act on.

    Returns:
        Our own messages, which explain themselves and never carry content. Anything
        else gives only its type name, since the message could hold a piece of the file.
    """
    if isinstance(exc, (ScanError, Blocked, MaskingError)):
        return str(exc)
    return f"rezunate-guard failed unexpectedly ({type(exc).__name__})"


def main() -> None:
    """Read a payload from stdin, print the reply, and always exit 0.

    Only exit 0 means "use what I printed". Any other code counts as a non-blocking
    error and the original tool output stands, so an unhandled exception here is a leak
    rather than a crash. That is why everything is caught.
    """
    reply = None
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            payload = {}
        reply = respond(payload)
    except BaseException as exc:  # noqa: BLE001 - an escaping exception is a leak
        if payload.get("hook_event_name") == "PreToolUse":
            # PreToolUse ignores a withheld reply and runs the call anyway, so denying is
            # the only way to fail closed here.
            reply = _deny(f"rezunate-guard could not vet this call: {_reason_for(exc)}")
        else:
            reply = _withhold(payload, _reason_for(exc))

    if reply is not None:
        sys.stdout.write(json.dumps(reply))
    sys.exit(0)


if __name__ == "__main__":
    main()
