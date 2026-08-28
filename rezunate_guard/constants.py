"""Names and paths shared by more than one module."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_FILENAME = ".rezunate-guard.yaml"

HOME_ENV = "REZUNATE_HOME"
API_KEY_ENV = "REZUNATE_LLM_API_KEY"
BASE_URL_ENV = "ROUTER_BASE_URL"
CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"


def rezunate_home() -> Path:
    """Return the directory holding our own state.

    Reads the environment on every call, so `REZUNATE_HOME` can redirect it. That is
    what keeps a test run out of the developer's real home directory.

    Returns:
        `$REZUNATE_HOME`, or `~/.rezunate` if it is unset.
    """
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".rezunate")


def credentials_path() -> Path:
    """Return the file holding the saved API key."""
    return rezunate_home() / "credentials"


def placeholder_key_path() -> Path:
    """Return the file holding the placeholder key."""
    return rezunate_home() / "placeholder.key"


def log_path() -> Path:
    """Return the file holding problems worth telling the user about."""
    return rezunate_home() / "guard.log"


def user_config_path() -> Path:
    """Return the config used for files that belong to no project."""
    return rezunate_home() / "config.yaml"
