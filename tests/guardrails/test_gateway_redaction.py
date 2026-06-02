"""End-to-end tests: redaction applied through chat_complete.

Uses the Anthropic provider because it speaks ``requests`` and is therefore
interceptable by the ``responses`` mock (the OpenAI provider uses the official
httpx-based SDK and bypasses it).
"""

import responses

from rezunate_llm_sdk.gateway import ChatCompletionRequest, chat_complete
from rezunate_llm_sdk.models import (
    GuardrailAction,
    GuardrailRule,
    GuardrailsConfig,
)

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _redact_ssn_config() -> GuardrailsConfig:
    return GuardrailsConfig(
        guardrails=[
            GuardrailRule(
                name="redact-ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.REDACT,
                replacement="[SSN]",
            )
        ]
    )


def _anthropic_response_with(text: str) -> dict:
    return {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


class TestOutputRedaction:
    @responses.activate
    def test_redacts_pii_in_llm_output(self, mock_api_key, sample_messages):
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("The SSN on file is 123-45-6789."),
            status=200,
        )

        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=sample_messages,
            max_tokens=100,
        )
        result = chat_complete(
            provider="anthropic",
            api_key=mock_api_key,
            request=request,
            guardrails_config=_redact_ssn_config(),
        )

        assert result.choices[0].message.content == "The SSN on file is [SSN]."

    @responses.activate
    def test_clean_output_is_untouched(self, mock_api_key, sample_messages):
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("Hello! How can I assist you today?"),
            status=200,
        )

        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=sample_messages,
            max_tokens=100,
        )
        result = chat_complete(
            provider="anthropic",
            api_key=mock_api_key,
            request=request,
            guardrails_config=_redact_ssn_config(),
        )

        assert result.choices[0].message.content == "Hello! How can I assist you today?"


class TestInputRedaction:
    @responses.activate
    def test_redacts_pii_before_sending_to_provider(self, mock_api_key):
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("ok"),
            status=200,
        )

        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
            max_tokens=100,
        )
        chat_complete(
            provider="anthropic",
            api_key=mock_api_key,
            request=request,
            guardrails_config=_redact_ssn_config(),
        )

        sent_body = responses.calls[0].request.body.decode("utf-8")
        assert "123-45-6789" not in sent_body
        assert "[SSN]" in sent_body
