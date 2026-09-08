"""Reversible PII masking: the vault, local-regex masking, and end-to-end
mask-in / rehydrate-out through Gateway.chat_complete.

Uses the Anthropic provider (interceptable by the ``responses`` mock) and mocks
the hosted guardrail scan (`/api/v1/guardrails/scan`).
"""

from concurrent.futures import ThreadPoolExecutor

import responses

import rezunate_llm_sdk.constants as constants
from rezunate_llm_sdk.gateway import ChatCompletionRequest, Gateway, chat_complete
from rezunate_llm_sdk.masking import MaskVault, unmask
from rezunate_llm_sdk.models import (
    ChatCompletionResponse,
    Choice,
    FinishReason,
    GuardrailAction,
    GuardrailDirection,
    GuardrailRule,
    GuardrailsConfig,
    ResponseMessage,
    Role,
    ServerGuardrailsConfig,
    Usage,
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
SCAN_URL = f"{constants.ROUTER_BASE_URL}/api/v1/guardrails/scan"
ROUTER_KEY = "rez-test-key"

NO_LOCAL_RULES = GuardrailsConfig(guardrails=[])
SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.-]+"


def _redact_config() -> GuardrailsConfig:
    return GuardrailsConfig(
        guardrails=[
            GuardrailRule(
                name="ssn",
                pattern=SSN_PATTERN,
                action=GuardrailAction.REDACT,
                replacement="[SSN]",
            ),
            GuardrailRule(
                name="email",
                pattern=EMAIL_PATTERN,
                action=GuardrailAction.REDACT,
                replacement="[EMAIL]",
            ),
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


def _scan_response(*, entities=None, action="none", blocked=False, text="") -> dict:
    return {"entities": entities or [], "action": action, "blocked": blocked, "text": text}


# --------------------------------------------------------------------------- #
# MaskVault unit behavior
# --------------------------------------------------------------------------- #


class TestMaskVault:
    def test_assign_is_value_consistent(self):
        vault = MaskVault()
        first = vault.assign("person name", "John")
        again = vault.assign("person name", "John")
        other = vault.assign("person name", "Jane")

        assert first == again == "[PERSON_NAME_1]"
        assert other == "[PERSON_NAME_2]"

    def test_restore_round_trips(self):
        vault = MaskVault()
        p1 = vault.assign("email address", "john@acme.com")
        p2 = vault.assign("email address", "jane@acme.com")
        masked = f"Draft to {p1} and cc {p2}."

        assert vault.restore(masked) == "Draft to john@acme.com and cc jane@acme.com."

    def test_mask_entities_from_spans(self):
        text = "My email is john@acme.com now"
        entities = [{"text": "john@acme.com", "label": "email address", "start": 12, "end": 25}]
        vault = MaskVault()
        masked = vault.mask_entities(text, entities)

        assert masked == "My email is [EMAIL_ADDRESS_1] now"
        assert vault.restore(masked) == text

    def test_dumps_loads_preserves_and_extends(self):
        vault = MaskVault()
        vault.assign("email address", "john@acme.com")

        reloaded = MaskVault.loads(vault.dumps())
        # Existing mapping restores...
        assert reloaded.restore("[EMAIL_ADDRESS_1]") == "john@acme.com"
        # ...same value reuses its placeholder, new value continues numbering.
        assert reloaded.assign("email address", "john@acme.com") == "[EMAIL_ADDRESS_1]"
        assert reloaded.assign("email address", "kate@acme.com") == "[EMAIL_ADDRESS_2]"

    def test_vaults_are_isolated(self):
        # Two end-users, same placeholder token, different underlying values.
        user_a = MaskVault()
        user_b = MaskVault()
        user_a.assign("email address", "a@acme.com")
        user_b.assign("email address", "b@acme.com")

        assert user_a.restore("[EMAIL_ADDRESS_1]") == "a@acme.com"
        assert user_b.restore("[EMAIL_ADDRESS_1]") == "b@acme.com"

    def test_unmask_helper(self):
        vault = MaskVault()
        p = vault.assign("phone", "555-111-2222")
        assert unmask(f"call {p}", vault) == "call 555-111-2222"


# --------------------------------------------------------------------------- #
# Local-regex masking via check_guardrails + backward-compat
# --------------------------------------------------------------------------- #


class TestLocalRegexReversible:
    def test_vault_produces_unique_reversible_placeholders(self):
        from rezunate_llm_sdk.guardrails import check_guardrails

        vault = MaskVault()
        text = "SSNs 111-22-3333 and 444-55-6666"
        redacted, _ = check_guardrails(text, _redact_config(), GuardrailDirection.INPUT, vault)

        assert redacted == "SSNs [SSN_1] and [SSN_2]"
        assert vault.restore(redacted) == text

    def test_without_vault_uses_static_replacement(self):
        # Backward-compat: default path unchanged, both matches collapse.
        from rezunate_llm_sdk.guardrails import check_guardrails

        text = "SSNs 111-22-3333 and 444-55-6666"
        redacted, _ = check_guardrails(text, _redact_config(), GuardrailDirection.INPUT)

        assert redacted == "SSNs [SSN] and [SSN]"


# --------------------------------------------------------------------------- #
# End-to-end through Gateway.chat_complete
# --------------------------------------------------------------------------- #


def _gateway(**kwargs) -> Gateway:
    return Gateway(
        default_provider="anthropic",
        default_api_key="anthropic-key",
        REZUNATE_LLM_API_KEY=ROUTER_KEY,
        **kwargs,
    )


class TestLocalRegexEndToEnd:
    @responses.activate
    def test_input_masked_and_output_rehydrated(self, mock_api_key):
        # Model echoes the placeholder it was given.
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("Your number [SSN_1] is on file."),
            status=200,
        )
        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
            max_tokens=100,
        )

        result = _gateway(guardrails_config=_redact_config()).chat_complete(
            request, reversible=True
        )

        # Provider never saw the raw SSN...
        sent = responses.calls[0].request.body.decode("utf-8")
        assert "123-45-6789" not in sent
        assert "[SSN_1]" in sent
        # ...but the caller gets it back.
        assert result.choices[0].message.content == "Your number 123-45-6789 is on file."

    @responses.activate
    def test_reversible_off_keeps_static_masking(self, mock_api_key):
        # Backward-compat: no reversible flag -> static token, no rehydrate.
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("noted [SSN]"),
            status=200,
        )
        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
            max_tokens=100,
        )

        result = _gateway(guardrails_config=_redact_config()).chat_complete(request)

        sent = responses.calls[0].request.body.decode("utf-8")
        assert "[SSN]" in sent and "[SSN_1]" not in sent
        assert result.choices[0].message.content == "noted [SSN]"

    @responses.activate
    def test_module_chat_complete_masks_input_without_rehydrating(self, mock_api_key):
        # Low-level function masks input into the vault; caller restores manually.
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("ok [SSN_1]"),
            status=200,
        )
        vault = MaskVault()
        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
            max_tokens=100,
        )

        result = chat_complete(
            provider="anthropic",
            api_key="anthropic-key",
            request=request,
            guardrails_config=_redact_config(),
            vault=vault,
        )

        # Output left masked by the low-level function...
        assert result.choices[0].message.content == "ok [SSN_1]"
        # ...vault holds the map for manual restore.
        assert unmask(result.choices[0].message.content, vault) == "ok 123-45-6789"

    @responses.activate
    def test_conversation_resume_keeps_stable_numbering(self, mock_api_key):
        # Turn 1: mask, persist the vault (dumps).
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("ok [SSN_1]"),
            status=200,
        )
        cfg = _redact_config()
        vault = MaskVault()
        _gateway(guardrails_config=cfg).chat_complete(
            ChatCompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "SSN 123-45-6789"}],
                max_tokens=100,
            ),
            reversible=True,
            vault=vault,
        )
        saved = vault.dumps()

        # Turn 2 (days later): reload; a new SSN continues at _2.
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("both [SSN_1] and [SSN_2]"),
            status=200,
        )
        resumed = MaskVault.loads(saved)
        result = _gateway(guardrails_config=cfg).chat_complete(
            ChatCompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "and 999-88-7777"}],
                max_tokens=100,
            ),
            reversible=True,
            vault=resumed,
        )

        assert result.choices[0].message.content == "both 123-45-6789 and 999-88-7777"


class TestServerScanEndToEnd:
    @responses.activate
    def test_server_input_masked_reversibly_and_rehydrated(self, mock_api_key):
        # Input scan detects the email via spans; output side scan is off so we
        # only exercise reversible input masking + rehydrate.
        responses.add(
            responses.POST,
            SCAN_URL,
            json=_scan_response(
                entities=[
                    {
                        "text": "john@acme.com",
                        "label": "email address",
                        "score": 0.99,
                        "start": 12,
                        "end": 25,
                    }
                ],
                action="redact",
                text="My email is [EMAIL].",
            ),
            status=200,
        )
        responses.add(
            responses.POST,
            ANTHROPIC_URL,
            json=_anthropic_response_with("Confirmed [EMAIL_ADDRESS_1]."),
            status=200,
        )

        gw = _gateway(
            guardrails_config=NO_LOCAL_RULES,
            server_guardrails=ServerGuardrailsConfig(directions=("input",)),
        )
        result = gw.chat_complete(
            ChatCompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "My email is john@acme.com"}],
                max_tokens=100,
            ),
            reversible=True,
        )

        # Provider got the placeholder, not the raw email.
        provider_body = responses.calls[1].request.body.decode("utf-8")
        assert "john@acme.com" not in provider_body
        assert "[EMAIL_ADDRESS_1]" in provider_body
        # Caller gets the real email back.
        assert result.choices[0].message.content == "Confirmed john@acme.com."


class TestStreamingRehydrate:
    def test_streamed_output_is_rehydrated(self, monkeypatch):
        # The Anthropic streaming path uses httpx (not interceptable by
        # `responses`), so feed the SSE lines in directly. The placeholder is
        # split across two deltas to exercise the partial-token carry.
        from rezunate_llm_sdk.providers.base import BaseProvider

        def _line(text):
            return (
                f'data: {{"type":"content_block_delta","index":0,'
                f'"delta":{{"type":"text_delta","text":"{text}"}}}}'
            )

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m","model":'
            '"claude-sonnet-4-20250514","usage":{"input_tokens":5,"output_tokens":0}}}',
            "",
            "event: content_block_delta",
            _line("Number [SSN"),
            "",
            "event: content_block_delta",
            _line("_1] noted."),
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":6}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]

        def fake_lines(self, url, body, headers=None):
            yield from sse_lines

        monkeypatch.setattr(BaseProvider, "_stream_http_lines", fake_lines)

        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "My SSN is 123-45-6789"}],
            max_tokens=100,
            stream=True,
        )
        stream = _gateway(guardrails_config=_redact_config()).chat_complete(
            request, reversible=True
        )

        text = "".join(
            c.delta.content
            for chunk in stream
            for c in chunk.choices
            if c.delta and c.delta.content
        )
        assert "123-45-6789" in text
        assert "[SSN_1]" not in text


class TestConcurrentClientIsolation:
    """One shared Gateway serving many end-clients at once (the B2B shape).

    Each request masks into its *own* vault, so numbering restarts every call:
    the provider should always receive ``[EMAIL_1]``/``[SSN_1]``. If a vault were
    shared on the Gateway, later clients would be sent ``[EMAIL_2]``, ``[EMAIL_3]``
    ... and cross-client value-consistency would leak — so asserting every
    provider-facing prompt uses ``_1`` numbering catches that regression.
    """

    def test_shared_gateway_isolates_concurrent_reversible_calls(self, monkeypatch):
        from rezunate_llm_sdk.providers.base import BaseProvider

        # Record what the provider actually receives (the masked prompt), then
        # echo it back — an "LLM" that repeats the placeholders it was sent.
        # list.append is atomic under the GIL, so this is thread-safe.
        seen_by_provider: list[str] = []

        def echo_complete(self, request):
            content = request.messages[-1].content
            seen_by_provider.append(content)
            return ChatCompletionResponse(
                id="echo",
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ResponseMessage(role=Role.ASSISTANT, content=content),
                        finish_reason=FinishReason.STOP,
                    )
                ],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        monkeypatch.setattr(BaseProvider, "chat_complete", echo_complete)

        gateway = _gateway(guardrails_config=_redact_config())  # one shared instance

        def run(i: int) -> tuple[str, str]:
            content = f"Contact user{i}@acme.com, SSN 123-45-{i:04d}."
            request = ChatCompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": content}],
                max_tokens=100,
            )
            result = gateway.chat_complete(request, reversible=True)
            return content, result.choices[0].message.content

        n = 200
        with ThreadPoolExecutor(max_workers=32) as pool:
            results = list(pool.map(run, range(n)))

        # 1) Every client gets back exactly its own values — correct round-trip.
        assert len(results) == n
        for original, restored in results:
            assert restored == original

        # 2) Isolation with teeth: because each call used a fresh vault, the
        # provider only ever saw first-position tokens. A shared/leaked vault
        # would send higher indices to later clients.
        assert len(seen_by_provider) == n
        for masked in seen_by_provider:
            assert "[EMAIL_1]" in masked and "[SSN_1]" in masked
            assert "[EMAIL_2]" not in masked and "[SSN_2]" not in masked
