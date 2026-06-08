"""Tests for regex guardrails: block, flag, and redact actions."""

import pytest

from rezunate_llm_sdk.guardrails import (
    GuardrailsError,
    check_guardrails,
)
from rezunate_llm_sdk.models import (
    GuardrailAction,
    GuardrailDirection,
    GuardrailRule,
    GuardrailsConfig,
)

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"


def _config(*rules: GuardrailRule) -> GuardrailsConfig:
    return GuardrailsConfig(guardrails=list(rules))


class TestCheckGuardrailsRedact:
    """Redaction behaviour for the ``redact`` action."""

    def test_redacts_match_with_default_replacement(self):
        config = _config(
            GuardrailRule(
                name="redact-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.REDACT,
            )
        )

        redacted, violations = check_guardrails(
            "My SSN is 123-45-6789.", config, GuardrailDirection.OUTPUT
        )

        assert redacted == "My SSN is [REDACTED]."
        assert len(violations) == 1
        assert violations[0].action == GuardrailAction.REDACT
        assert violations[0].match == "123-45-6789"

    def test_redacts_with_custom_replacement(self):
        config = _config(
            GuardrailRule(
                name="redact-email",
                pattern=EMAIL_PATTERN,
                action=GuardrailAction.REDACT,
                replacement="[EMAIL]",
            )
        )

        redacted, _ = check_guardrails(
            "Reach me at alex@example.com please.",
            config,
            GuardrailDirection.OUTPUT,
        )

        assert redacted == "Reach me at [EMAIL] please."

    def test_redacts_all_occurrences(self):
        config = _config(
            GuardrailRule(
                name="redact-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.REDACT,
            )
        )

        redacted, _ = check_guardrails(
            "111-22-3333 and 444-55-6666", config, GuardrailDirection.OUTPUT
        )

        assert redacted == "[REDACTED] and [REDACTED]"

    def test_multiple_redact_rules_apply_together(self):
        config = _config(
            GuardrailRule(
                name="redact-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.REDACT,
                replacement="[SSN]",
            ),
            GuardrailRule(
                name="redact-email",
                pattern=EMAIL_PATTERN,
                action=GuardrailAction.REDACT,
                replacement="[EMAIL]",
            ),
        )

        redacted, violations = check_guardrails(
            "ssn 123-45-6789 email a@b.com", config, GuardrailDirection.OUTPUT
        )

        assert redacted == "ssn [SSN] email [EMAIL]"
        assert len(violations) == 2

    def test_no_match_returns_text_unchanged(self):
        config = _config(
            GuardrailRule(
                name="redact-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.REDACT,
            )
        )

        text = "Nothing sensitive here."
        redacted, violations = check_guardrails(text, config, GuardrailDirection.OUTPUT)

        assert redacted == text
        assert violations == []


class TestCheckGuardrailsBlockFlag:
    """Block and flag actions keep their existing behaviour."""

    def test_block_raises(self):
        config = _config(
            GuardrailRule(
                name="block-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.BLOCK,
            )
        )

        with pytest.raises(GuardrailsError) as exc_info:
            check_guardrails("SSN 123-45-6789", config, GuardrailDirection.OUTPUT)

        assert exc_info.value.rule_name == "block-ssn"
        assert exc_info.value.direction == GuardrailDirection.OUTPUT

    def test_flag_records_violation_without_changing_text(self):
        config = _config(
            GuardrailRule(
                name="flag-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.FLAG,
            )
        )

        text = "SSN 123-45-6789"
        redacted, violations = check_guardrails(text, config, GuardrailDirection.OUTPUT)

        assert redacted == text
        assert len(violations) == 1
        assert violations[0].action == GuardrailAction.FLAG


class TestCheckGuardrailsReturn:
    """check_guardrails returns a (redacted_text, violations) tuple."""

    def test_returns_text_and_violations_tuple(self):
        config = _config(
            GuardrailRule(
                name="flag-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.FLAG,
            )
        )

        result = check_guardrails("SSN 123-45-6789", config, GuardrailDirection.INPUT)

        assert isinstance(result, tuple)
        redacted, violations = result
        assert redacted == "SSN 123-45-6789"
        assert isinstance(violations, list)
        assert len(violations) == 1
