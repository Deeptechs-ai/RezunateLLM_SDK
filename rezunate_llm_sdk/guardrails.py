"""Guardrails - Regex-based content filtering for inputs and outputs."""

import re
from pathlib import Path

import yaml

from rezunate_llm_sdk.masking import MaskVault
from rezunate_llm_sdk.models import (
    DetectedEntity,
    GuardrailAction,
    GuardrailDirection,
    GuardrailsConfig,
    GuardrailViolation,
)


class GuardrailsError(Exception):
    """Raised when content matches a local regex guardrail rule."""

    def __init__(
        self, rule_name: str, rule_description: str, direction: GuardrailDirection
    ) -> None:
        self.rule_name = rule_name
        self.rule_description = rule_description
        self.direction = direction
        super().__init__(
            f"Guardrail '{rule_name}' triggered on {direction.name}: {rule_description}"
        )


class ServerGuardrailsError(Exception):
    """Raised when the server-side PII scan blocks content.

    Attributes:
        direction: Whether the blocked content was input or output.
        entities: PII entities the scan detected.
        action: The action label the server returned (e.g. "block").
    """

    def __init__(
        self,
        direction: GuardrailDirection,
        entities: list[DetectedEntity],
        action: str,
    ) -> None:
        self.direction = direction
        self.entities = entities
        self.action = action
        labels = ", ".join(sorted({e.label for e in entities})) or "PII"
        super().__init__(f"Server guardrail blocked {direction.name}: detected {labels}")


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
    direction: GuardrailDirection,
    vault: MaskVault | None = None,
) -> tuple[str, list[GuardrailViolation]]:
    """Check text against all guardrail rules and apply redactions.

    Rules are evaluated in order. ``block`` rules raise immediately, ``flag``
    rules only record a violation, and ``redact`` rules replace every match
    with the rule's ``replacement`` text in the returned string.

    Args:
        text: The text to check.
        config: Guardrails configuration with rules.
        direction: Either "input" or "output", for error reporting.
        vault: Optional masking vault. With one, ``redact`` rules emit unique,
            restorable placeholders instead of the static ``replacement``.
            Without one (default), redaction stays one-way as before.

    Returns:
        A tuple of ``(redacted_text, violations)``. ``redacted_text`` equals
        ``text`` when no ``redact`` rule matched.

    Raises:
        GuardrailsError: If any violation has action="block".
    """
    violations: list[GuardrailViolation] = []
    redacted = text

    for rule in config.guardrails:
        match = re.search(rule.pattern, text)
        if not match:
            continue

        violations.append(
            GuardrailViolation(
                rule_name=rule.name,
                rule_description=rule.description,
                direction=direction,
                action=rule.action,
                match=match.group(0),
            )
        )

        if rule.action == GuardrailAction.BLOCK:
            raise GuardrailsError(
                rule_name=rule.name,
                rule_description=rule.description,
                direction=direction,
            )

        if rule.action == GuardrailAction.REDACT:
            if vault is not None:
                # Bind ``label`` as a default so the substitution callback does
                # not capture the loop variable.
                redacted = re.sub(
                    rule.pattern,
                    lambda m, label=rule.name: vault.assign(label, m.group(0)),
                    redacted,
                )
            else:
                redacted = re.sub(rule.pattern, rule.replacement, redacted)

    return redacted, violations
