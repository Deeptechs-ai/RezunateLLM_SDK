"""
Tests for Grok (xAI) Provider.

Grok uses xAI's OpenAI-compatible API, so GrokProvider subclasses
OpenAIProvider and only overrides base_url, provider_name, and endpoint.
Tests here focus on that wiring; OpenAI-shape passthrough behaviour is
already covered by test_openai_provider.py.
"""

from unittest.mock import MagicMock

from rezunate_llm_sdk.models import ChatCompletionRequest, ChatCompletionResponse, Provider
from rezunate_llm_sdk.providers.grok_provider import GrokProvider


class TestGrokProviderProperties:
    """Tests for Grok provider properties."""

    def test_base_url(self, mock_api_key):
        provider = GrokProvider(api_key=mock_api_key)
        assert provider.base_url == "https://api.x.ai/v1"

    def test_client_base_url_matches(self, mock_api_key):
        """The OpenAI SDK client must be initialized with the xAI base URL."""
        provider = GrokProvider(api_key=mock_api_key)
        assert str(provider.client.base_url).rstrip("/") == "https://api.x.ai/v1"

    def test_provider_name(self, mock_api_key):
        provider = GrokProvider(api_key=mock_api_key)
        assert provider.provider_name == Provider.GROK

    def test_endpoint(self, mock_api_key):
        provider = GrokProvider(api_key=mock_api_key)
        assert provider.get_endpoint() == "/chat/completions"

    def test_headers(self, mock_api_key):
        """Headers exist for interface consistency even when SDK ignores them."""
        provider = GrokProvider(api_key=mock_api_key)
        headers = provider.get_headers()
        assert headers["Authorization"] == f"Bearer {mock_api_key}"
        assert headers["Content-Type"] == "application/json"


class TestGrokTransformRequest:
    """GrokProvider inherits passthrough from OpenAIProvider."""

    def test_request_passthrough(self, mock_api_key, openai_request):
        provider = GrokProvider(api_key=mock_api_key)
        req_obj = ChatCompletionRequest.model_validate(openai_request)

        result = provider.transform_request(req_obj)

        assert result is req_obj


def _fake_sdk_response(payload: dict) -> MagicMock:
    """Return a mock that mimics the OpenAI SDK response object."""
    fake = MagicMock()
    fake.model_dump.return_value = payload
    return fake


class TestGrokSDKForwarding:
    """Verify Grok-specific SDK call behaviour (not covered by OpenAI tests)."""

    def test_xai_specific_extras_forwarded(self, mock_api_key, mocker, grok_response):
        """xAI's ``search_parameters`` extra survives the passthrough into the SDK."""
        provider = GrokProvider(api_key=mock_api_key)
        mock_create = mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(grok_response),
        )

        search_params = {"mode": "auto", "max_search_results": 5}
        provider.chat_complete(
            ChatCompletionRequest.model_validate(
                {
                    "model": "grok-3-mini",
                    "messages": [{"role": "user", "content": "What's the news?"}],
                    "search_parameters": search_params,
                }
            )
        )

        assert mock_create.call_args.kwargs["search_parameters"] == search_params

    def test_multi_turn_conversation_forwarded(
        self, mock_api_key, mocker, grok_response, sample_conversation
    ):
        """All four turns survive intact in the SDK call."""
        provider = GrokProvider(api_key=mock_api_key)
        mock_create = mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(grok_response),
        )

        provider.chat_complete(
            ChatCompletionRequest.model_validate(
                {"model": "grok-3-mini", "messages": sample_conversation}
            )
        )

        roles = [m["role"] for m in mock_create.call_args.kwargs["messages"]]
        assert roles == ["system", "user", "assistant", "user"]


class TestGrokResponseHandling:
    """Verify response shape and finish-reason mapping for Grok-tagged responses."""

    def test_provider_field_set_to_grok(self, mock_api_key, mocker, grok_response):
        """Response carries ``provider == Provider.GROK`` even though shape is OpenAI's."""
        provider = GrokProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(grok_response),
        )

        result = provider.chat_complete(
            ChatCompletionRequest(
                model="grok-3-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        )

        assert result.provider == Provider.GROK

    def test_finish_reason_length_mapped(self, mock_api_key, mocker, grok_response):
        truncated = dict(grok_response)
        truncated["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Truncated..."},
                "finish_reason": "length",
            }
        ]
        provider = GrokProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(truncated),
        )

        result = provider.chat_complete(
            ChatCompletionRequest(
                model="grok-3-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        )

        assert result.choices[0].finish_reason == "length"

    def test_finish_reason_tool_calls_mapped(self, mock_api_key, mocker, grok_response):
        tool_call = dict(grok_response)
        tool_call["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "calling tool"},
                "finish_reason": "tool_calls",
            }
        ]
        provider = GrokProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(tool_call),
        )

        result = provider.chat_complete(
            ChatCompletionRequest(
                model="grok-3-mini",
                messages=[{"role": "user", "content": "Hi"}],
            )
        )

        assert result.choices[0].finish_reason == "tool_calls"


class TestGrokIntegration:
    """End-to-end tests with the OpenAI SDK client mocked."""

    def test_full_chat_completion(self, mock_api_key, mocker, grok_response):
        """Mock the OpenAI SDK and verify chat_complete wires through correctly."""
        provider = GrokProvider(api_key=mock_api_key)

        mock_create = mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(grok_response),
        )

        req = ChatCompletionRequest(
            model="grok-3-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        result = provider.chat_complete(req)

        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["model"] == "grok-3-mini"

        assert isinstance(result, ChatCompletionResponse)
        assert result.provider == Provider.GROK
        assert result.choices[0].message.content == "Hello! How can I assist you today?"
        assert result.usage.total_tokens == 30

    def test_error_returned_as_error_info(self, mock_api_key, mocker):
        """API errors are caught and returned as ChatCompletionResponse.error."""
        provider = GrokProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            side_effect=RuntimeError("Invalid API key"),
        )

        req = ChatCompletionRequest(
            model="grok-3-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        result = provider.chat_complete(req)

        assert result.error is not None
        assert "Invalid API key" in result.error.message
        assert result.provider == Provider.GROK
