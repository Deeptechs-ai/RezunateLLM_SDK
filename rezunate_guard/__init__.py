"""rezunate-guard — redact PII from files before Claude Code sends them to the model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rezunate-llm-sdk")
except PackageNotFoundError:  # running from a source tree, not an install
    __version__ = "0.0.0+dev"
