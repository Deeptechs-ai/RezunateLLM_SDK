"""Guardrails - Regex-based content filtering for inputs and outputs."""

import re
from pathlib import Path

import yaml

from llm_router.models import GuardrailsConfig


class GuardrailsError(Exception):
    """Raised when content matches a guardrail rule."""

    def __init__(self, rule_name: str, rule_description: str, direction: str) -> None:
        self.rule_name = rule_name
        self.rule_description = rule_description
        self.direction = direction
        super().__init__(f"Guardrail '{rule_name}' triggered on {direction}: {rule_description}")


def load_guardrails(config_path: str | Path) -> GuardrailsConfig:
    """Load guardrails configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated GuardrailsConfig with pre-compiled regex patterns.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or patterns fail to compile.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Guardrails config not found: {path}")

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "guardrails" not in data:
        raise ValueError("Guardrails config must contain a 'guardrails' key")

    config = GuardrailsConfig.model_validate(data)

    # Pre-compile patterns to catch invalid regex early
    for rule in config.guardrails:
        try:
            re.compile(rule.pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex in rule '{rule.name}': {e}") from e

    return config


def check_guardrails(
    text: str,
    config: GuardrailsConfig,
    direction: str,
) -> None:
    """Check text against all guardrail rules.

    Args:
        text: The text to check.
        config: Guardrails configuration with rules.
        direction: Either "input" or "output", for error reporting.

    Raises:
        GuardrailsError: If text matches any guardrail rule.
    """
    for rule in config.guardrails:
        if re.search(rule.pattern, text):
            raise GuardrailsError(
                rule_name=rule.name,
                rule_description=rule.description,
                direction=direction,
            )
