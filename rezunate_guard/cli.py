"""The `rezunate-guard` command: login, init, install, uninstall, status, check."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rezunate_guard import __version__, constants
from rezunate_guard.config import CONFIG_TEMPLATE, resolve_config, scan_decision
from rezunate_guard.hook import main as hook_main
from rezunate_guard.scanner import stored_key

#: PostToolUse redacts what came back. PreToolUse refuses reads whose contents would
#: never return as text we could redact.
HOOK_EVENTS = ("PreToolUse", "PostToolUse")

#: Every tool
HOOK_MATCHER = "*"

#: Identifies our entries, so uninstall never disturbs hooks someone else registered.
HOOK_MARKERS = ("rezunate_guard", "rezunate-guard")


def hook_command() -> str:
    """Return the command to register, bound to this interpreter.

    Not the bare name. Installed as a library the script lands in a venv's `bin`, which
    Claude Code may not have on PATH, and a hook it cannot launch leaves the file
    unredacted.
    """
    return f"{sys.executable} -m rezunate_guard hook"


def settings_path(scope: str) -> Path:
    """Return the Claude Code settings file for a scope.

    Args:
        scope: Either "user" or "project".

    Returns:
        The settings path. Project scope is relative to the working directory.
    """
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    return (
        Path(os.environ.get(constants.CLAUDE_CONFIG_DIR_ENV) or Path.home() / ".claude")
        / "settings.json"
    )


def _read_json(path: Path) -> dict:
    """Read a JSON object from a file, or return {} if it is missing or unreadable."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    """Write JSON atomically, merging is the caller's job.

    `settings.json` usually holds unrelated settings, so an interrupted write must not
    corrupt it.

    The temporary file carries our pid, as in `masking.placeholder_key`. Two processes
    sharing one scratch name can interleave a write and a rename, leaving a half-written
    file in place of the user's settings.

    Args:
        path: The file to write.
        data: What to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _entry_is_ours(entry: object) -> bool:
    """Return True if a hook entry was registered by us, so uninstall leaves others be."""
    if not isinstance(entry, dict):
        return False
    return any(
        isinstance(hook, dict)
        and any(marker in str(hook.get("command", "")) for marker in HOOK_MARKERS)
        for hook in entry.get("hooks", [])
    )


def _installed_events(path: Path) -> list[str]:
    """Return which of our hook events are registered in a settings file."""
    hooks = _read_json(path).get("hooks")
    if not isinstance(hooks, dict):
        return []
    return [
        event
        for event in HOOK_EVENTS
        if isinstance(hooks.get(event), list) and any(map(_entry_is_ours, hooks[event]))
    ]


def command_login(args: argparse.Namespace) -> int:
    """Save an API key to `~/.rezunate/credentials`, readable only by the user.

    Prompts for the key when it is not given as an argument.
    """
    key = (args.key or "").strip()
    if not key:
        print(f"No key yet? Create one at {constants.api_keys_url()}")
        try:
            key = input("API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled", file=sys.stderr)
            return 1

    if not key:
        print(f"no key given; create one at {constants.api_keys_url()}", file=sys.stderr)
        return 1

    path = constants.credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")

    print(f"key saved to {path}")
    print("this key is not verified until the first scan; run `rezunate-guard status`")
    return 0


def command_init(args: argparse.Namespace) -> int:
    """Write a config template into the current directory.

    The template is fully commented out, so a fresh config protects nothing until the
    user edits it.
    """
    path = Path.cwd() / constants.CONFIG_FILENAME
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite", file=sys.stderr)
        return 1

    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    print(f"wrote {path}")
    print()
    print("Nothing is protected yet. Edit the `scan:` list, then run:")
    print("  rezunate-guard status")
    return 0


def _allow_redacted_copies(settings: dict) -> bool:
    """Let Claude Code open the redacted copies.

    They sit outside every project so they are never scanned themselves, which also puts
    them outside Claude Code's working directories.

    Returns:
        True if the entry was added, False if it was already there.

    Raises:
        ValueError: If `permissions` is shaped in a way we should not overwrite.
    """
    permissions = settings.setdefault("permissions", {})
    if not isinstance(permissions, dict):
        raise ValueError("a 'permissions' key that is not an object")

    directories = permissions.setdefault("additionalDirectories", [])
    if not isinstance(directories, list):
        raise ValueError("a malformed 'permissions.additionalDirectories' list")

    copies_dir = str(constants.redacted_copies_dir())
    if copies_dir in directories:
        return False

    directories.append(copies_dir)
    return True


def _disallow_redacted_copies(settings: dict) -> bool:
    """Drop our directory from `permissions`, leaving anything else there untouched."""
    permissions = settings.get("permissions")
    if not isinstance(permissions, dict):
        return False

    directories = permissions.get("additionalDirectories")
    if not isinstance(directories, list):
        return False

    copies_dir = str(constants.redacted_copies_dir())
    if copies_dir not in directories:
        return False

    remaining = [entry for entry in directories if entry != copies_dir]
    if remaining:
        permissions["additionalDirectories"] = remaining
    else:
        del permissions["additionalDirectories"]
    if not permissions:
        del settings["permissions"]
    return True


def command_install(args: argparse.Namespace) -> int:
    """Register both hook events in `settings.json`, adding whichever is missing."""
    path = settings_path(args.scope)
    settings = _read_json(path)

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        print(f"{path} has a 'hooks' key that is not an object; fix it first", file=sys.stderr)
        return 1

    changes: list[str] = []
    for event in HOOK_EVENTS:
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            print(f"{path} has a malformed {event} list; fix it first", file=sys.stderr)
            return 1

        if any(_entry_is_ours(entry) for entry in entries):
            continue

        entries.append(
            {"matcher": HOOK_MATCHER, "hooks": [{"type": "command", "command": hook_command()}]}
        )
        changes.append(f"{event}: registered for {HOOK_MATCHER!r}")

    try:
        if _allow_redacted_copies(settings):
            changes.append(f"permissions: allowed reads of {constants.redacted_copies_dir()}")
    except ValueError as exc:
        print(f"{path} has {exc}; fix it first", file=sys.stderr)
        return 1

    # The directory has to exist for Claude Code to accept it as a working directory.
    constants.redacted_copies_dir().mkdir(parents=True, exist_ok=True, mode=0o700)

    if not changes:
        print(f"already installed in {path}")
        return 0

    _write_json(path, settings)
    for change in changes:
        print(f"  {change}")
    print(f"hook installed in {path}")
    print("Restart Claude Code, or start a new session, for it to take effect.")
    return 0


def command_uninstall(args: argparse.Namespace) -> int:
    """Remove our hook entries from `settings.json`, leaving any others untouched."""
    path = settings_path(args.scope)
    settings = _read_json(path)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        if _disallow_redacted_copies(settings):
            _write_json(path, settings)
            print(f"redacted-copies directory removed from {path}")
            return 0
        print(f"nothing to remove in {path}")
        return 0

    removed = _disallow_redacted_copies(settings)
    for event in HOOK_EVENTS:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue

        remaining = [entry for entry in entries if not _entry_is_ours(entry)]
        if len(remaining) == len(entries):
            continue

        removed = True
        if remaining:
            hooks[event] = remaining
        else:
            del hooks[event]

    if not removed:
        print(f"nothing to remove in {path}")
        return 0

    if not hooks:
        del settings["hooks"]

    _write_json(path, settings)
    print(f"hook removed from {path}")
    print("Files are no longer redacted. Restart Claude Code to apply.")
    return 0


def _report_key() -> list[str]:
    """Print the API key line of `status`.

    Returns:
        Problems found, for the caller to summarise. Empty when the key is fine.
    """
    if os.environ.get(constants.API_KEY_ENV, "").strip():
        print(f"  key        set via {constants.API_KEY_ENV}")
    elif stored_key():
        print(f"  key        saved in {constants.credentials_path()}")
    else:
        print("  key        MISSING")
        return ["No API key. Run `rezunate-guard login`. Protected files will be withheld."]
    return []


def _report_hook() -> list[str]:
    """Print the hook lines of `status`.

    Returns:
        Problems found. A half-registered hook counts as one.
    """
    installed = {
        path: events
        for path in (settings_path("user"), settings_path("project"))
        if (events := _installed_events(path))
    }

    if not installed:
        print("  hook       NOT INSTALLED")
        return [
            "The hook is not registered. Run `rezunate-guard install`. Nothing is being redacted."
        ]

    problems = []
    for path, events in installed.items():
        print(f"  hook       installed in {path} ({', '.join(events)})")
        missing = [event for event in HOOK_EVENTS if event not in events]
        if missing:
            problems.append(
                f"Only part of the hook is registered in {path}; {', '.join(missing)} "
                "is missing. Run `rezunate-guard install`."
            )
    return problems


def _report_config() -> list[str]:
    """Print the config lines of `status`.

    Returns:
        Problems found. A config protecting nothing counts as one.
    """
    # resolve_config takes the file being read, so ask about a placeholder name in the
    # working directory.
    config = resolve_config(Path.cwd() / "x")

    if config.source is None:
        print("  config     none found")
        return [
            f"No {constants.CONFIG_FILENAME} in this directory or its parents. "
            "Run `rezunate-guard init`. Nothing is being scanned."
        ]

    print(f"  config     {config.source}")

    if config.error:
        print(f"  ERROR      {config.error}")
        return [f"The config could not be read: {config.error}. Nothing is being scanned."]

    if config.protects_nothing:
        return [
            "The config lists no folders to scan, so nothing is protected. "
            f"Add folders under `scan:` in {config.source}."
        ]

    missing = []
    for directory in config.scan:
        exists = directory.is_dir()
        print(f"  scan       {directory}{'' if exists else '   MISSING'}")
        if not exists:
            missing.append(directory)

    if missing:
        return [
            f"{len(missing)} listed folder(s) do not exist, so nothing in them is "
            f"protected: {', '.join(str(path) for path in missing)}."
        ]
    return []


def _report_redacted_copies() -> list[str]:
    """Print whether Claude Code may open the redacted copies.

    Returns:
        Problems found. Without the entry a PDF is extracted and redacted correctly and
        the read is then refused, which reads as the file being unsupported.
    """
    copies_dir = str(constants.redacted_copies_dir())
    for path in (settings_path("user"), settings_path("project")):
        permissions = _read_json(path).get("permissions")
        if not isinstance(permissions, dict):
            continue
        directories = permissions.get("additionalDirectories")
        if isinstance(directories, list) and copies_dir in directories:
            print(f"  copies     readable, listed in {path}")
            return []

    print("  copies     NOT READABLE")
    return [
        f"Claude Code cannot open the redacted copies in {copies_dir}, so PDFs and Word "
        "files will be refused after being redacted. Run `rezunate-guard install`."
    ]


def command_status(args: argparse.Namespace) -> int:
    """Report whether the guard would actually protect anything right now.

    A working guard is invisible, so a broken one is too: no key, no config, a config
    protecting nothing, a half-registered hook. Each fails silently in normal use, so
    this names them and exits non-zero.
    """
    print(f"rezunate-guard {__version__}")
    print()

    problems = _report_key() + _report_hook() + _report_config() + _report_redacted_copies()
    if not problems:
        print()
        print("Protection is active.")
        return 0

    print()
    print("Not protecting anything yet:" if len(problems) > 1 else "One thing to fix:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


def command_check(args: argparse.Namespace) -> int:
    """Print whether each given path would be scanned, and why."""
    for name in args.paths:
        path = Path(name).expanduser()
        decision = scan_decision(path)
        print(f"{'scan' if decision.should_scan else 'skip':5} {path}  ({decision.reason})")
    return 0


def command_hook(args: argparse.Namespace) -> int:
    """Run the hook. Claude Code calls this, not people."""
    hook_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="rezunate-guard",
        description="Redact PII from files before Claude Code sends them to the model.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    login = subcommands.add_parser("login", help="save your RezunateLLM API key")
    login.add_argument("key", nargs="?", help="read from a prompt if omitted")
    login.set_defaults(func=command_login)

    init = subcommands.add_parser("init", help=f"create {constants.CONFIG_FILENAME} here")
    init.add_argument("--force", action="store_true", help="overwrite an existing config")
    init.set_defaults(func=command_init)

    install = subcommands.add_parser("install", help="register the hook with Claude Code")
    install.add_argument(
        "--scope",
        choices=("user", "project"),
        default="user",
        help="user (default) applies everywhere; project applies to this directory only",
    )
    install.set_defaults(func=command_install)

    uninstall = subcommands.add_parser("uninstall", help="unregister the hook")
    uninstall.add_argument("--scope", choices=("user", "project"), default="user")
    uninstall.set_defaults(func=command_uninstall)

    status = subcommands.add_parser("status", help="check whether protection is actually on")
    status.set_defaults(func=command_status)

    check = subcommands.add_parser("check", help="show whether given paths would be scanned")
    check.add_argument("paths", nargs="+")
    check.set_defaults(func=command_check)

    hook = subcommands.add_parser("hook", help=argparse.SUPPRESS)
    hook.set_defaults(func=command_hook)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments to parse. Reads `sys.argv` when omitted.

    Returns:
        The process exit code.
    """
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
