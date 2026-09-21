"""Server-side PII guardrails applied to LLM output via Gateway.chat_complete.

Mocks two endpoints: the Anthropic provider (LLM output) and the hosted
guardrail scan (`/api/v1/guardrails/scan`).
"""

import pytest
import responses

import rezunate_llm_sdk.constants as constants
from rezunate_llm_sdk.gateway import ChatCompletionRequest, Gateway
from rezunate_llm_sdk.guardrails import ServerGuardrailsError
from rezunate_llm_sdk.models import GuardrailsConfig, ServerGuardrailsConfig

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
SCAN_URL = f"{constants.REZUNATE_LLM_BASE_URL}/api/v1/guardrails/scan"
ROUTER_KEY = "rez-test-key"

# Explicit empty config so these server-side tests are isolated from any local
# regex rules auto-loaded via the GUARDRAILS_FILE_PATH env var.
NO_LOCAL_RULES = GuardrailsConfig(guardrails=[])


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


def _scan_response(*, entities=None, action="none", blocked=False, text="") -> dict:
    return {
        "entities": entities or [],
        "action": action,
        "blocked": blocked,
        "text": text,
    }


def _gateway(server_guardrails=True) -> Gateway:
    return Gateway(
        default_provider="anthropic",
        default_api_key="anthropic-key",
        guardrails_config=NO_LOCAL_RULES,
        server_guardrails=server_guardrails,
        REZUNATE_LLM_API_KEY=ROUTER_KEY,
    )


# Scan only the output side, for tests that target output behaviour in isolation.
OUTPUT_ONLY = ServerGuardrailsConfig(directions=("output",))


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Tell me about Maria."}],
        max_tokens=100,
    )


class TestServerOutputRedaction:
    @responses.activate
    def test_redacts_output_using_server_text(self, mock_api_key):
        raw = "Her email is maria@example.com."
        responses.add(responses.POST, ANTHROPIC_URL, json=_anthropic_response_with(raw), status=200)
        responses.add(
            responses.POST,
            SCAN_URL,
            json=_scan_response(
                entities=[
                    {
                        "text": "maria@example.com",
                        "label": "EMAIL",
                        "score": 0.99,
                        "start": 13,
                        "end": 30,
                    }
                ],
                action="redact",
                blocked=False,
                text="Her email is [EMAIL].",
            ),
            status=200,
        )

        result = _gateway(OUTPUT_ONLY).chat_complete(_request())

        assert result.choices[0].message.content == "Her email is [EMAIL]."
        # Provider call + one (output) scan call.
        assert len(responses.calls) == 2

    @responses.activate
    def test_blocks_output_when_scan_blocks(self, mock_api_key):
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("The SSN is 123-45-6789."),
            status=200,
        )
        responses.add(
            responses.POST,
            SCAN_URL,
            json=_scan_response(
                entities=[
                    {
                        "text": "123-45-6789",
                        "label": "US_SSN",
                        "score": 0.98,
                        "start": 11,
                        "end": 22,
                    }
                ],
                action="block",
                blocked=True,
                text="The SSN is 123-45-6789.",
            ),
            status=200,
        )

        with pytest.raises(ServerGuardrailsError) as exc_info:
            _gateway().chat_complete(_request())

        assert "US_SSN" in str(exc_info.value)
        assert exc_info.value.action == "block"

    @responses.activate
    def test_clean_output_passes_through(self, mock_api_key):
        clean = "Maria is a fictional sales rep."
        responses.add(
            responses.POST, ANTHROPIC_URL, json=_anthropic_response_with(clean), status=200
        )
        responses.add(responses.POST, SCAN_URL, json=_scan_response(text=clean), status=200)

        result = _gateway().chat_complete(_request())

        assert result.choices[0].message.content == clean

    @responses.activate
    def test_disabled_by_default_skips_scan(self, mock_api_key):
        raw = "Her email is maria@example.com."
        responses.add(responses.POST, ANTHROPIC_URL, json=_anthropic_response_with(raw), status=200)

        gateway = Gateway(
            default_provider="anthropic",
            default_api_key="anthropic-key",
            guardrails_config=NO_LOCAL_RULES,
            REZUNATE_LLM_API_KEY=ROUTER_KEY,
        )
        result = gateway.chat_complete(_request())

        # No scan endpoint registered; only the provider was called.
        assert result.choices[0].message.content == raw
        assert len(responses.calls) == 1

    @responses.activate
    def test_per_call_override_enables_scan(self, mock_api_key):
        raw = "Her email is maria@example.com."
        responses.add(responses.POST, ANTHROPIC_URL, json=_anthropic_response_with(raw), status=200)
        responses.add(
            responses.POST,
            SCAN_URL,
            json=_scan_response(action="redact", text="Her email is [EMAIL]."),
            status=200,
        )

        gateway = Gateway(
            default_provider="anthropic",
            default_api_key="anthropic-key",
            guardrails_config=NO_LOCAL_RULES,
            REZUNATE_LLM_API_KEY=ROUTER_KEY,
        )
        result = gateway.chat_complete(_request(), server_guardrails=True)

        assert result.choices[0].message.content == "Her email is [EMAIL]."


class TestServerDirections:
    """`directions` chooses which side(s) the scan runs on; True = both."""

    @responses.activate
    def test_true_scans_both_input_and_output(self, mock_api_key):
        responses.add(
            responses.POST, ANTHROPIC_URL, json=_anthropic_response_with("ok"), status=200
        )
        responses.add(
            responses.POST,
            SCAN_URL,
            json=_scan_response(action="redact", text="[CLEAN]"),
            status=200,
        )

        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "My email is maria@example.com"}],
            max_tokens=100,
        )
        _gateway(server_guardrails=True).chat_complete(request)

        # Provider + input scan + output scan = 3 calls.
        scan_calls = [c for c in responses.calls if "guardrails/scan" in c.request.url]
        assert len(scan_calls) == 2

    @responses.activate
    def test_input_only_scans_prompt_not_output(self, mock_api_key):
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("Output email bob@example.com"),
            status=200,
        )
        responses.add(
            responses.POST,
            SCAN_URL,
            json=_scan_response(action="redact", text="My email is [EMAIL]"),
            status=200,
        )

        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "My email is maria@example.com"}],
            max_tokens=100,
        )
        result = _gateway(ServerGuardrailsConfig(directions=("input",))).chat_complete(request)

        # Prompt was scanned/redacted before sending; output left untouched.
        provider_call = next(c for c in responses.calls if "anthropic" in c.request.url)
        assert "maria@example.com" not in provider_call.request.body.decode("utf-8")
        assert result.choices[0].message.content == "Output email bob@example.com"
        scan_calls = [c for c in responses.calls if "guardrails/scan" in c.request.url]
        assert len(scan_calls) == 1
