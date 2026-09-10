"""Decide whether a file sits in a protected folder, and so needs scanning."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from rezunate_guard import constants, log

CONFIG_TEMPLATE = """\
# rezunate-guard — nothing is scanned until you list it here.
#
# List the folders holding personal data. Everything inside one is redacted before
# Claude Code sends it to the model. Paths are relative to this file, and an absolute
# path works too. Each scanned file is a billable API call, so list only what needs it.
scan:
  # - clients
  # - data/patients
"""


@dataclass(frozen=True)
class Decision:
    """Whether to scan a file, and why.

    Attributes:
        should_scan: True if the file's contents need scanning.
        reason: Why, for `status` and `check` to print.
    """

    should_scan: bool
    reason: str


@dataclass(frozen=True)
class GuardConfig:
    """A parsed config file.

    Attributes:
        root: The folder this config sits in.
        scan: Protected folders, as absolute paths.
        source: The file this came from, if any.
        error: Why the config couldn't be read. Nothing is protected while it is set.
    """

    root: Path
    scan: tuple[Path, ...] = ()
    source: Path | None = None
    error: str | None = None

    @property
    def protects_nothing(self) -> bool:
        """True when no folder is listed.

        Worth surfacing: a guard that quietly does nothing is worse than no guard.
        """
        return not self.scan


def is_protected(path: Path, directory: Path) -> bool:
    """Check whether a path is a protected folder or anything inside it.

    Both must be resolved already, or a symlink makes this answer no when it should
    answer yes.
    """
    return path.is_relative_to(directory)


def broken_config(root: Path, source: Path | None, error: str) -> GuardConfig:
    """Build the config used when one exists but can't be read.

    Scans nothing, the error is logged and shown by `status`,
    which is the only thing that tells the user they are unprotected.

    Args:
        root: The folder the config was found in.
        source: The config file, if there was one.
        error: What went wrong.

    Returns:
        A config protecting nothing, carrying the error.
    """
    log.problem(source or root, error)
    return GuardConfig(root=root, source=source, error=error)


def _resolve_folder(name: str, root: Path) -> Path:
    """Turn one entry from `scan:` into an absolute folder.

    A relative name hangs off `root`, the folder the config sits in. An absolute or `~`
    name is taken as written, so a config can protect a folder outside its own project.

    Args:
        name: One entry from the `scan:` list, already stripped of surrounding space.
        root: The folder the config sits in.

    Returns:
        The folder, absolute and resolved.

    Raises:
        ValueError: If the name is a pattern, or `~` points nowhere. Either resolves to a
            folder that never exists, leaving the user believing they were covered.
    """
    text = name.rstrip("/") or "/"
    if any(character in text for character in "*?["):
        raise ValueError(f"{name!r} is not a folder; name the folder, such as 'clients'")

    try:
        folder = Path(text).expanduser()
    except RuntimeError as exc:
        raise ValueError(f"{name!r} is not a folder: {exc}") from exc

    return (folder if folder.is_absolute() else root / folder).resolve()


def _listed_folders(value: object, root: Path) -> tuple[Path, ...]:
    """Return the folders listed under `scan:`.

    Args:
        value: The raw `scan` value: one folder name, or a list of them.
        root: The folder the config sits in.

    Returns:
        Absolute, resolved folders.

    Raises:
        ValueError: If an entry is not a folder name. See `_resolve_folder`.
    """
    entries = [value] if isinstance(value, str) else value
    if not isinstance(entries, list | tuple):
        entries = []

    folders = []
    for entry in entries:
        if entry is not None and not isinstance(entry, str):
            raise ValueError(f"{entry!r} is not a folder name; check the punctuation")

        name = (entry or "").strip()
        if name:
            folders.append(_resolve_folder(name, root))
    return tuple(folders)


def find_config_file(start: Path) -> Path | None:
    """Walk up from a path to the nearest config file.

    Args:
        start: The file being read, not the working directory.

    Returns:
        The nearest config file, or None. Nearest wins, so one folder in a monorepo can
        protect itself more tightly than the repo around it.
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for directory in [current, *current.parents]:
        candidate = directory / constants.CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


#: Configs already parsed, keyed by the file and its contents. Reading is cheap, parsing
#: is not, and keying on the text cannot serve a stale answer the way a timestamp can.
_PARSED: dict[tuple[Path, str], GuardConfig] = {}


def load_config(config_path: Path) -> GuardConfig:
    """Parse a config file, reusing the last parse of the same contents.

    One tool call asks about every path it touched, and they usually share a config, so
    the file was being parsed once per path.

    Never raises. Anything it can't read comes back as a `broken_config`, which protects
    nothing and logs why.

    Args:
        config_path: The file to parse.

    Returns:
        The parsed config, or a broken one carrying the error.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return broken_config(config_path.parent, config_path, f"could not read config: {exc}")

    key = (config_path, text)
    if key not in _PARSED:
        _PARSED[key] = _parse(config_path, text)
    return _PARSED[key]


def _parse_scan_list(text: str, root: Path) -> tuple[Path, ...]:
    """Read the `scan:` list out of a config file's text.

    Args:
        text: The file's contents.
        root: The folder the config sits in, which relative entries hang off.

    Returns:
        The folders listed, as absolute paths.

    Raises:
        ValueError: If the file cannot be understood.
        OSError: If a listed folder cannot be resolved.
    """
    try:
        raw = yaml.safe_load(text) or {}
    except Exception as exc:  # noqa: BLE001 - every failure has to be reported, not raised
        raise ValueError(f"could not read config: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    unknown = sorted(set(raw) - {"scan"})
    if unknown:
        raise ValueError(f"unknown setting {unknown[0]!r}; expected `scan:`")

    return _listed_folders(raw.get("scan"), root)


def _parse(config_path: Path, text: str) -> GuardConfig:
    """Turn the contents of one config file into rules. See `load_config`, which caches this."""
    root = config_path.parent

    try:
        scan = _parse_scan_list(text, root)
    except (ValueError, OSError) as exc:
        return broken_config(root, config_path, str(exc))

    if not scan:
        log.problem(config_path, "no folders listed under `scan:`")

    # A folder that isn't there is how a mis-indented list shows up: the YAML parses,
    # and the entry quietly matches nothing.
    for directory in scan:
        if not directory.is_dir():
            log.problem(config_path, f"folder not found: {directory}")

    return GuardConfig(root=root, scan=scan, source=config_path)


def resolve_config(file_path: Path) -> GuardConfig:
    """Find and load the config that governs a path.

    Falls back to `~/.rezunate/config.yaml` for files in no project. That one is rooted
    at the filesystem root, so it is the last place to look and its folders should be
    absolute.

    Args:
        file_path: The file being read.

    Returns:
        The config governing it. Protects nothing if there is none.
    """
    config_file = find_config_file(file_path)
    if config_file is not None:
        return load_config(config_file)

    filesystem_root = Path(file_path.anchor or "/")
    fallback = constants.user_config_path()
    if fallback.is_file():
        return dataclasses.replace(load_config(fallback), root=filesystem_root)

    return GuardConfig(root=filesystem_root)


def scan_decision(file_path: Path | str, config: GuardConfig | None = None) -> Decision:
    """Decide whether a file needs scanning.

    Nothing is scanned until the user lists a folder, since every scan is billable. The
    answer comes from the file's own path, never the working directory, because
    `claude --add-dir` reaches files belonging to another project.

    Args:
        file_path: The file being read.
        config: The config to judge it by. Resolved from the file's own path if omitted.

    Returns:
        The decision, with a reason the user can read.
    """
    path = Path(file_path).resolve()
    if config is None:
        config = resolve_config(path)

    # Listed folders come first: they may sit outside the project, and checking the root
    # first would hand such a file to a config that has never heard of it.
    for directory in config.scan:
        if is_protected(path, directory):
            return Decision(True, f"in a protected folder ({directory})")

    if not path.is_relative_to(config.root.resolve()):
        # Almost certainly an --add-dir file, so let its own config answer.
        own_config = resolve_config(path)
        if own_config.root != config.root:
            decision = scan_decision(path, own_config)
            return Decision(decision.should_scan, f"outside config root; {decision.reason}")
        return Decision(False, "outside config root and no config of its own")

    reason = (
        "no folders configured for scanning"
        if config.protects_nothing
        else "not in a protected folder"
    )
    return Decision(False, reason)


def should_scan(file_path: Path | str, config: GuardConfig | None = None) -> bool:
    """Return just the answer from `scan_decision`, for callers that don't need the reason."""
    return scan_decision(file_path, config).should_scan
