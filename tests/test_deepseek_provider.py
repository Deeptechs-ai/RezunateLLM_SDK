"""
Tests for DeepSeek Provider.

DeepSeek uses an OpenAI-compatible API, so DeepSeekProvider subclasses
OpenAIProvider and only overrides base_url, provider_name, and endpoint.
Tests here focus on that wiring; OpenAI-shape passthrough behaviour is
already covered by test_openai_provider.py.
"""

from unittest.mock import MagicMock

from rezunate_llm_sdk.models import ChatCompletionRequest, ChatCompletionResponse, Provider
from rezunate_llm_sdk.providers.deepseek_provider import DeepSeekProvider


class TestDeepSeekProviderProperties:
    """Tests for DeepSeek provider properties."""

    def test_base_url(self, mock_api_key):
        provider = DeepSeekProvider(api_key=mock_api_key)
        assert provider.base_url == "https://api.deepseek.com/v1"

    def test_client_base_url_matches(self, mock_api_key):
        """The OpenAI SDK client must be initialized with DeepSeek's base URL."""
        provider = DeepSeekProvider(api_key=mock_api_key)
        assert str(provider.client.base_url).rstrip("/") == "https://api.deepseek.com/v1"

    def test_provider_name(self, mock_api_key):
        provider = DeepSeekProvider(api_key=mock_api_key)
        assert provider.provider_name == Provider.DEEPSEEK

    def test_endpoint(self, mock_api_key):
        provider = DeepSeekProvider(api_key=mock_api_key)
        assert provider.get_endpoint() == "/chat/completions"

    def test_headers(self, mock_api_key):
        provider = DeepSeekProvider(api_key=mock_api_key)
        headers = provider.get_headers()
        assert headers["Authorization"] == f"Bearer {mock_api_key}"
        assert headers["Content-Type"] == "application/json"


class TestDeepSeekTransformRequest:
    """DeepSeekProvider inherits passthrough from OpenAIProvider."""

    def test_request_passthrough(self, mock_api_key, openai_request):
        provider = DeepSeekProvider(api_key=mock_api_key)
        req_obj = ChatCompletionRequest.model_validate(openai_request)

        result = provider.transform_request(req_obj)

        assert result is req_obj


def _fake_sdk_response(payload: dict) -> MagicMock:
    """Return a mock that mimics the OpenAI SDK response object."""
    fake = MagicMock()
    fake.model_dump.return_value = payload
    return fake


class TestDeepSeekSDKForwarding:
    """Verify DeepSeek-specific SDK call behaviour (not covered by OpenAI tests)."""

    def test_deepseek_prefix_extra_forwarded(self, mock_api_key, mocker, deepseek_response):
        """DeepSeek's beta ``prefix`` field (prefix completion) reaches the SDK call."""
        provider = DeepSeekProvider(api_key=mock_api_key)
        mock_create = mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(deepseek_response),
        )

        provider.chat_complete(
            ChatCompletionRequest.model_validate(
                {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "Continue: Hello,"}],
                    "prefix": "Hello, ",
                }
            )
        )

        assert mock_create.call_args.kwargs["prefix"] == "Hello, "

    def test_multi_turn_conversation_forwarded(
        self, mock_api_key, mocker, deepseek_response, sample_conversation
    ):
        """All four turns survive intact in the SDK call."""
        provider = DeepSeekProvider(api_key=mock_api_key)
        mock_create = mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(deepseek_response),
        )

        provider.chat_complete(
            ChatCompletionRequest.model_validate(
                {"model": "deepseek-chat", "messages": sample_conversation}
            )
        )

        roles = [m["role"] for m in mock_create.call_args.kwargs["messages"]]
        assert roles == ["system", "user", "assistant", "user"]


class TestDeepSeekResponseHandling:
    """Verify response shape and finish-reason mapping for DeepSeek-tagged responses."""

    def test_provider_field_set_to_deepseek(self, mock_api_key, mocker, deepseek_response):
        """Response carries ``provider == Provider.DEEPSEEK`` even though shape is OpenAI's."""
        provider = DeepSeekProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(deepseek_response),
        )

        result = provider.chat_complete(
            ChatCompletionRequest(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Hi"}],
            )
        )

        assert result.provider == Provider.DEEPSEEK

    def test_finish_reason_length_mapped(self, mock_api_key, mocker, deepseek_response):
        truncated = dict(deepseek_response)
        truncated["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Truncated..."},
                "finish_reason": "length",
            }
        ]
        provider = DeepSeekProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(truncated),
        )

        result = provider.chat_complete(
            ChatCompletionRequest(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Hi"}],
            )
        )

        assert result.choices[0].finish_reason == "length"

    def test_finish_reason_tool_calls_mapped(self, mock_api_key, mocker, deepseek_response):
        tool_call = dict(deepseek_response)
        tool_call["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "calling tool"},
                "finish_reason": "tool_calls",
            }
        ]
        provider = DeepSeekProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(tool_call),
        )

        result = provider.chat_complete(
            ChatCompletionRequest(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Hi"}],
            )
        )

        assert result.choices[0].finish_reason == "tool_calls"

    def test_insufficient_system_resource_mapped(self, mock_api_key, mocker, deepseek_response):
        """DeepSeek-specific finish reason for deepseek-reasoner under load.

        Maps to STOP via the dedicated entry in FINISH_REASON_MAP. This finish
        reason is not emitted by any other provider — DeepSeek-unique behaviour.
        """
        resource_limit = dict(deepseek_response)
        resource_limit["model"] = "deepseek-reasoner"
        resource_limit["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Partial reasoning..."},
                "finish_reason": "insufficient_system_resource",
            }
        ]
        provider = DeepSeekProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(resource_limit),
        )

        result = provider.chat_complete(
            ChatCompletionRequest(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "Hi"}],
            )
        )

        assert result.choices[0].finish_reason == "stop"


class TestDeepSeekIntegration:
    """End-to-end tests with the OpenAI SDK client mocked."""

    def test_full_chat_completion(self, mock_api_key, mocker, deepseek_response):
        """Mock the OpenAI SDK and verify chat_complete wires through correctly."""
        provider = DeepSeekProvider(api_key=mock_api_key)

        mock_create = mocker.patch.object(
            provider.client.chat.completions,
            "create",
            return_value=_fake_sdk_response(deepseek_response),
        )

        req = ChatCompletionRequest(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        result = provider.chat_complete(req)

        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["model"] == "deepseek-chat"

        assert isinstance(result, ChatCompletionResponse)
        assert result.provider == Provider.DEEPSEEK
        assert result.choices[0].message.content == "Hello! How can I assist you today?"
        assert result.usage.total_tokens == 30

    def test_error_returned_as_error_info(self, mock_api_key, mocker):
        """API errors are caught and returned as ChatCompletionResponse.error."""
        provider = DeepSeekProvider(api_key=mock_api_key)
        mocker.patch.object(
            provider.client.chat.completions,
            "create",
            side_effect=RuntimeError("Insufficient Balance"),
        )

        req = ChatCompletionRequest(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        result = provider.chat_complete(req)

        assert result.error is not None
        assert "Insufficient Balance" in result.error.message
        assert result.provider == Provider.DEEPSEEK
