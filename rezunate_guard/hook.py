"""The hook: the last point where PII can be stopped before the model sees it."""

from __future__ import annotations

import codecs
import glob
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, NamedTuple

from rezunate_guard import extract, log, redacted_copy
from rezunate_guard.config import should_scan
from rezunate_guard.extract import ExtractionError
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

#: One argument of a command, stopping at whatever would start a new one.
_COMMAND_TOKEN = re.compile(r"[^\s;|&()<>\"'`]+")

#: A trailing extension, so `notes.md` reads as a file but `grep` does not.
_LOOKS_LIKE_FILE = re.compile(r"\.\w{1,8}$")

#: Commands that report on a file without printing what is inside it. `ls`, `stat`, `du`
#: and `file` describe it; `basename`, `dirname`, `realpath` and `readlink` only rework
#: the path; `test` answers yes or no.
METADATA_COMMANDS = frozenset(
    {"basename", "dirname", "du", "file", "ls", "readlink", "realpath", "stat", "test"}
)

#: Where one command ends and the next begins, so each part is checked on its own.
_SHELL_SPLIT = re.compile(r"[;|&\n]+")


class Blocked(Exception):
    """The workspace guardrail blocks this content rather than redacting it."""


class RedactedCopy(NamedTuple):
    """A stand-in for a file whose contents the model cannot be shown.

    Attributes:
        path: The copy to read instead, or None if none could be made.
        reason: Why none could be made, for the refusal to quote. Empty on success.
    """

    path: Path | None
    reason: str = ""


class GivenFile(NamedTuple):
    """The file a tool was handed, and the key it arrived under.

    Attributes:
        key: The `tool_input` key naming it, so a redirect can rewrite that key.
        path: Its absolute path. Empty, with `key`, when the tool was given no file.
    """

    key: str
    path: str


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
    redacted = redact_all(pending) if pending else {}

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

        refers_to_a_file = (
            bool(token)
            and not token.startswith("-")
            and "://" not in token
            and ("/" in token or bool(_LOOKS_LIKE_FILE.search(token)))
        )
        if refers_to_a_file:
            found.append(token)

    return found


def _reads_no_contents(command: str) -> bool:
    """Return True if every part of a command reports on files without opening them.

    `ls` on a protected PDF prints a size, not a page of it. Refusing it told the model
    the file was off limits, so it never tried the Read that would have worked.

    Anything unrecognised, or able to hide a second command, is left to the refusal path.
    """
    parts: list[list[str]] = []
    has_shell_syntax = any(token in command for token in ("$(", "`", "<", ">"))
    if not has_shell_syntax:
        try:
            parts = [shlex.split(segment) for segment in _SHELL_SPLIT.split(command)]
        except ValueError:
            parts = []

    return bool(parts) and all(part and part[0] in METADATA_COMMANDS for part in parts)


def _tool_path(payload: dict[str, Any]) -> str:
    """Return the path the tool was pointed at, or "" if it named no file."""
    tool_input = payload.get("tool_input")
    fields = tool_input if isinstance(tool_input, dict) else {}

    named = [fields.get(key) for key in PATH_KEYS]
    return next((path for path in named if isinstance(path, str) and path), "")


def _given_file(payload: dict[str, Any]) -> GivenFile:
    """Return the file the tool was given, and the key it came under.

    Only a file handed to the tool directly can be swapped for a redacted copy. We can't
    swap a path inside a shell command without changing what the command does.

    Returns:
        The file, with empty fields if the tool was given none.
    """
    tool_input = payload.get("tool_input")
    fields = tool_input if isinstance(tool_input, dict) else {}

    cwd = payload.get("cwd")
    base = cwd if isinstance(cwd, str) else os.getcwd()

    return next(
        (
            GivenFile(key, os.path.join(base, os.path.expanduser(value)))
            for key in PATH_KEYS
            if isinstance(value := fields.get(key), str) and value
        ),
        GivenFile("", ""),
    )


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
    matches = sorted(glob.glob(path)) if any(character in path for character in "*?[") else []
    return [path, *matches]


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

    readable = b"\x00" not in sample
    try:
        # An incremental decoder forgives a character cut in half by the sample boundary,
        # which a plain decode would call binary.
        codecs.getincrementaldecoder("utf-8")().decode(sample)
    except UnicodeDecodeError:
        readable = False

    return readable


def _deny(reason: str) -> dict[str, Any]:
    """Build the PreToolUse reply that refuses a call outright."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _read_instead(updated_input: dict[str, Any], reason: str) -> dict[str, Any]:
    """Build the PreToolUse reply that points the call at a different file."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason,
            "updatedInput": updated_input,
        }
    }


def _redacted_copy_of(path: str) -> RedactedCopy:
    """Build a redacted copy of a file whose contents the model cannot be shown.

    Args:
        path: The file itself.

    Returns:
        The copy, or the reason none could be made. Callers quote that reason in their
        refusal, so a scan that was down never reads as a file we cannot handle.
    """
    copy = RedactedCopy(None, "there is no text extractor for this kind of file")

    if extract.can_extract(path):
        try:
            text = extract.extract_text(path)
            copy_path = redacted_copy.write(path, redact_all([text])[text])
            copy = RedactedCopy(copy_path)
        except (ExtractionError, ScanError, Blocked, MaskingError, OSError) as exc:
            log.problem(path, f"could not build a redacted copy: {exc}")
            copy = RedactedCopy(None, str(exc))

    return copy


def _refusal(path: str, copy: RedactedCopy) -> str:
    """Say why a call was refused, and where to go instead."""
    name = os.path.basename(path)
    if copy.path is not None:
        reason = (
            f"{name} is protected, so this call cannot run: its contents would reach the "
            f"model where rezunate-guard cannot rewrite them. Read {copy.path} instead. It "
            "holds the same text with the personal data replaced."
        )
    else:
        reason = (
            f"{name} is protected and rezunate-guard could not build a redacted copy of it: "
            f"{copy.reason}. None of its contents are available; do not guess at them, "
            "and do not change the `scan:` list to work around this. "
            "Ask the user to run `rezunate-guard status`."
        )

    return reason


def _before_tool(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Handle calls whose content could not be redacted afterwards.

    A file the tool was handed directly is swapped for a redacted copy where one can be
    made. Everything else is refused, since its content would reach the model somewhere
    we never see.

    Returns:
        A redirect or deny reply, or None to let the call run untouched.
    """
    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    reads_no_contents = isinstance(command, str) and _reads_no_contents(command)

    # A protected file whose text we have to pull out ourselves, because what the tool
    # returns for it is not text that PostToolUse could rewrite.
    needs_extraction = (
        path
        for path in _implicated_paths(payload)
        if os.path.isfile(path)
        and should_scan(path)
        and (extract.can_extract(path) or not _reaches_the_model_as_text(path))
    )
    file_to_extract = None if reads_no_contents else next(needs_extraction, None)

    reply = None
    if file_to_extract is not None:
        given_file = _given_file(payload)
        copy = _redacted_copy_of(file_to_extract)

        # realpath rather than samefile, which raises when the tool named a path that is
        # not there. Both sides go through it, so a symlink still matches its target.
        redirect_to_copy = (
            copy.path is not None
            and bool(given_file.key)
            and os.path.realpath(given_file.path) == os.path.realpath(file_to_extract)
        )
        if redirect_to_copy:
            reply = _read_instead(
                {**payload["tool_input"], given_file.key: str(copy.path)},
                f"This read returned the full text of {os.path.basename(file_to_extract)}, "
                "with personal data replaced by placeholders. Nothing failed and nothing is "
                "missing beyond those values: the file is protected, and no other way of "
                "reading it will return more.",
            )
        else:
            reply = _deny(_refusal(file_to_extract, copy))

    return reply


def _rewrite_strings(value: Any, transform, field: str | None = None) -> Any:
    """Copy a value, rewriting every content string with `transform`.

    Walks the whole response, so it works for any tool without knowing its schema. The
    shape is kept, since Claude Code drops a reply that doesn't match.

    Args:
        value: Any part of a tool response.
        transform: Turns a content string into its replacement.
        field: The key `value` sits under, used to spot structure.

    Returns:
        The rewritten copy.
    """
    if isinstance(value, str):
        # Leave structure alone: empty strings, paths, and "type" tags the schema checks.
        is_content = (
            bool(value)
            and field not in PATH_VALUED_KEYS
            and not (field == "type" and _TAG_VALUE.fullmatch(value))
        )
        rewritten = transform(value) if is_content else value
    elif isinstance(value, dict):
        rewritten = {name: _rewrite_strings(item, transform, name) for name, item in value.items()}
    elif isinstance(value, list):
        # List items have no key, so they inherit the parent's.
        rewritten = [_rewrite_strings(item, transform, field) for item in value]
    else:
        rewritten = value

    return rewritten


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
    fields = response if isinstance(response, dict) else {}

    return fields.get("type") == "image" or fields.get("isImage") is True


def _content_is_elsewhere(response: Any) -> bool:
    """Return True for a file result holding no text to rewrite.

    A PDF is the usual case: its pages are counted, but its text never appears.
    """
    fields = response if isinstance(response, dict) else {}
    file = fields.get("file")

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

    read_shaped = (
        isinstance(response, dict)
        and response.get("type") == "text"
        and isinstance(response.get("file"), dict)
    )

    if read_shaped:
        reply = _replace_content(payload, notice)
    elif _is_image(payload) and path:
        # Blanking the base64 in place would leave an image-shaped result holding
        # nothing, so send a text response instead.
        reply = _reply(_as_text_response(path, notice))
    elif not _holds_content(response):
        # Asked before blanking, not discovered afterwards. Falling through on a response
        # that did hold content would leave the original standing.
        reply = None
    else:
        placed = False

        def blank(text: str) -> str:
            nonlocal placed
            takes_notice = not placed and bool(text.strip())
            placed = placed or takes_notice

            return notice if takes_notice else ""

        reply = _reply(_rewrite_strings(response, blank))

    return reply


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
    response = payload.get("tool_response")

    # Short-circuits, so a PreToolUse call never pays for expanding the paths it touched.
    has_protected_output = (
        event == "PostToolUse"
        and response is not None
        and any(should_scan(path) for path in _implicated_paths(payload))
    )

    if event == "PreToolUse":
        reply = _before_tool(payload)
    elif not has_protected_output:
        reply = None
    elif _is_image(payload):
        reply = _withhold(payload, "images cannot be redacted")
    elif _content_is_elsewhere(response):
        # PreToolUse should have refused this. If it is not installed, withholding is all
        # that is left, since the content reaches the model by a route we never see.
        reply = _withhold(payload, "this result holds content we cannot rewrite")
    else:
        reply = _reply(redact_response(response))

    return reply


def _reason_for(exc: BaseException) -> str:
    """Turn an exception into a reason the user can act on.

    Returns:
        Our own messages, which explain themselves and never carry content. Anything
        else gives only its type name, since the message could hold a piece of the file.
    """
    ours = isinstance(exc, ScanError | Blocked | MaskingError)
    return str(exc) if ours else f"rezunate-guard failed unexpectedly ({type(exc).__name__})"


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
