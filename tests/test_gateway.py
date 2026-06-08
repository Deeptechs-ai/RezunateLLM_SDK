"""
Tests for Gateway (main router).
"""

from unittest.mock import MagicMock

import pytest
import responses

from rezunate_llm_sdk.gateway import (
    ChatCompletionRequest,
    Gateway,
    chat_complete,
    get_available_providers,
)


QWEN_FULL_URL = (
    "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)
LLAMA_FULL_URL = "https://api.llama.com/v1/chat/completions"


def _patch_openai_class_in(provider_module: str, mocker, response_payload: dict):
    """Patch the OpenAI class imported by an OpenAI-compat provider module.

    Returns the mocked ``chat.completions.create`` callable so tests can
    assert on call args. Used by Grok and DeepSeek gateway tests since the
    OpenAI Python SDK uses ``httpx`` and is not interceptable by the
    ``responses`` library.
    """
    fake_response = MagicMock()
    fake_response.model_dump.return_value = response_payload
    mock_openai_cls = mocker.patch(f"{provider_module}.OpenAI")
    mock_create = mock_openai_cls.return_value.chat.completions.create
    mock_create.return_value = fake_response
    return mock_create


class TestChatComplete:
    """Tests for chat_complete function."""

    @responses.activate
    def test_routes_to_openai(self, mock_api_key, sample_messages, openai_response):
        """Test routing to OpenAI provider."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        request = ChatCompletionRequest(model="gpt-4", messages=sample_messages)
        result = chat_complete(
            provider="openai",
            api_key=mock_api_key,
            request=request,
        )

        assert result.provider == "openai"
        assert len(responses.calls) == 1

    @responses.activate
    def test_routes_to_anthropic(self, mock_api_key, sample_messages, anthropic_response):
        """Test routing to Anthropic provider."""
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json=anthropic_response,
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
        )

        assert result.provider == "anthropic"
        assert len(responses.calls) == 1

    @responses.activate
    def test_routes_to_google(self, mock_api_key, sample_messages, google_response):
        """Test routing to Google provider."""
        responses.add(
            responses.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=google_response,
            status=200,
        )

        request = ChatCompletionRequest(model="gemini-2.0-flash", messages=sample_messages)
        result = chat_complete(
            provider="google",
            api_key=mock_api_key,
            request=request,
        )

        assert result.provider == "google"
        assert len(responses.calls) == 1

    def test_routes_to_grok(self, mock_api_key, mocker, sample_messages, grok_response):
        """Test routing to Grok (xAI) provider via Gateway-level chat_complete()."""
        mock_create = _patch_openai_class_in(
            "rezunate_llm_sdk.providers.grok_provider", mocker, grok_response
        )

        request = ChatCompletionRequest(model="grok-3-mini", messages=sample_messages)
        result = chat_complete(
            provider="grok",
            api_key=mock_api_key,
            request=request,
        )

        assert result.provider == "grok"
        mock_create.assert_called_once()

    def test_routes_to_deepseek(self, mock_api_key, mocker, sample_messages, deepseek_response):
        """Test routing to DeepSeek provider via Gateway-level chat_complete()."""
        mock_create = _patch_openai_class_in(
            "rezunate_llm_sdk.providers.deepseek_provider", mocker, deepseek_response
        )

        request = ChatCompletionRequest(model="deepseek-chat", messages=sample_messages)
        result = chat_complete(
            provider="deepseek",
            api_key=mock_api_key,
            request=request,
        )

        assert result.provider == "deepseek"
        mock_create.assert_called_once()

    @responses.activate
    def test_routes_to_qwen(self, mock_api_key, sample_messages, qwen_response):
        """Test routing to Qwen provider (native DashScope endpoint, HTTP-mocked)."""
        responses.add(responses.POST, QWEN_FULL_URL, json=qwen_response, status=200)

        request = ChatCompletionRequest(model="qwen-plus", messages=sample_messages)
        result = chat_complete(
            provider="qwen",
            api_key=mock_api_key,
            request=request,
        )

        assert result.provider == "qwen"
        assert len(responses.calls) == 1

    @responses.activate
    def test_routes_to_llama(self, mock_api_key, sample_messages, llama_response):
        """Test routing to Llama (Meta) provider (native API, HTTP-mocked)."""
        responses.add(responses.POST, LLAMA_FULL_URL, json=llama_response, status=200)

        request = ChatCompletionRequest(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=sample_messages,
        )
        result = chat_complete(
            provider="llama",
            api_key=mock_api_key,
            request=request,
        )

        assert result.provider == "llama"
        assert len(responses.calls) == 1

    @responses.activate
    def test_passes_temperature(self, mock_api_key, sample_messages, openai_response):
        """Test temperature is passed to provider."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=sample_messages,
            temperature=0.5,
        )
        chat_complete(
            provider="openai",
            api_key=mock_api_key,
            request=request,
        )

        import json

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["temperature"] == 0.5

    @responses.activate
    def test_passes_max_tokens(self, mock_api_key, sample_messages, openai_response):
        """Test max_tokens is passed to provider."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=sample_messages,
            max_tokens=100,
        )
        chat_complete(
            provider="openai",
            api_key=mock_api_key,
            request=request,
        )

        import json

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["max_tokens"] == 100

    @responses.activate
    def test_passes_additional_params(self, mock_api_key, sample_messages, openai_response):
        """Test additional params are passed to provider."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=sample_messages,
            top_p=0.9,
            frequency_penalty=0.5,
        )
        chat_complete(
            provider="openai",
            api_key=mock_api_key,
            request=request,
        )

        import json

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["top_p"] == 0.9
        assert request_body["frequency_penalty"] == 0.5

    def test_unknown_provider_raises_error(self, mock_api_key, sample_messages):
        """Test unknown provider raises ValueError."""
        request = ChatCompletionRequest(model="some-model", messages=sample_messages)
        with pytest.raises(ValueError) as exc_info:
            chat_complete(
                provider="unknown",
                api_key=mock_api_key,
                request=request,
            )

        assert "Unknown provider" in str(exc_info.value)

    @responses.activate
    def test_returns_openai_format(self, mock_api_key, sample_messages, anthropic_response):
        """Test response is in OpenAI format regardless of provider."""
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json=anthropic_response,
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
        )

        # Verify OpenAI format (now using Pydantic model)
        assert result.id is not None
        assert result.object == "chat.completion"
        assert result.choices is not None
        assert result.usage is not None
        assert result.usage.prompt_tokens is not None
        assert result.usage.completion_tokens is not None
        assert result.usage.total_tokens is not None


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""

    def test_returns_list(self):
        """Test returns a list."""
        providers = get_available_providers()
        assert isinstance(providers, list)

    def test_contains_all_providers(self):
        """Test contains all default providers."""
        providers = get_available_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "grok" in providers
        assert "llama" in providers
        assert "deepseek" in providers
        assert "qwen" in providers

    def test_returns_at_least_three_providers(self):
        """Test returns at least three providers."""
        providers = get_available_providers()
        assert len(providers) >= 3

    def test_returns_all_seven_providers(self):
        """Test that the registry now exposes all seven providers."""
        providers = get_available_providers()
        assert len(providers) == 7


class TestGatewayClass:
    """Tests for Gateway class."""

    def test_init_with_defaults(self, mock_api_key):
        """Test Gateway initializes with defaults."""
        gateway = Gateway(
            default_provider="openai",
            default_api_key=mock_api_key,
        )

        assert gateway.default_provider == "openai"
        assert gateway.default_api_key == mock_api_key

    def test_init_without_defaults(self):
        """Test Gateway initializes without defaults."""
        gateway = Gateway()

        assert gateway.default_provider is None
        assert gateway.default_api_key is None

    @responses.activate
    def test_chat_complete_uses_defaults(self, mock_api_key, sample_messages, openai_response):
        """Test chat_complete uses default provider and api_key."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        gateway = Gateway(
            default_provider="openai",
            default_api_key=mock_api_key,
        )

        request = ChatCompletionRequest(model="gpt-4", messages=sample_messages)
        result = gateway.chat_complete(request)

        assert result.provider == "openai"

    @responses.activate
    def test_chat_complete_overrides_defaults(
        self, mock_api_key, sample_messages, anthropic_response
    ):
        """Test chat_complete can override defaults."""
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json=anthropic_response,
            status=200,
        )

        gateway = Gateway(
            default_provider="openai",
            default_api_key="default-key",
        )

        request = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=sample_messages,
            max_tokens=100,
        )
        result = gateway.chat_complete(
            request,
            provider="anthropic",
            api_key=mock_api_key,
        )

        assert result.provider == "anthropic"

    def test_chat_complete_requires_provider(self, mock_api_key, sample_messages):
        """Test chat_complete raises error if no provider."""
        gateway = Gateway(default_api_key=mock_api_key)

        request = ChatCompletionRequest(model="gpt-4", messages=sample_messages)
        with pytest.raises(ValueError) as exc_info:
            gateway.chat_complete(request)

        assert "Provider must be specified" in str(exc_info.value)

    def test_chat_complete_requires_api_key(self, sample_messages):
        """Test chat_complete raises error if no api_key."""
        gateway = Gateway(default_provider="openai")

        request = ChatCompletionRequest(model="gpt-4", messages=sample_messages)
        with pytest.raises(ValueError) as exc_info:
            gateway.chat_complete(request)

        assert "API key must be specified" in str(exc_info.value)

    def test_providers_property(self):
        """Test providers property returns available providers."""
        gateway = Gateway()
        providers = gateway.providers

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "grok" in providers
        assert "llama" in providers
        assert "deepseek" in providers
        assert "qwen" in providers

    def test_gateway_with_grok_as_default(
        self, mock_api_key, mocker, sample_messages, grok_response
    ):
        """Gateway works when Grok is configured as the default provider."""
        _patch_openai_class_in("rezunate_llm_sdk.providers.grok_provider", mocker, grok_response)

        gateway = Gateway(default_provider="grok", default_api_key=mock_api_key)
        request = ChatCompletionRequest(model="grok-3-mini", messages=sample_messages)
        result = gateway.chat_complete(request)

        assert result.provider == "grok"
        assert result.choices[0].message.content == "Hello! How can I assist you today?"

    @responses.activate
    def test_gateway_with_qwen_as_default(self, mock_api_key, sample_messages, qwen_response):
        """Gateway works when Qwen (native) is configured as the default provider."""
        responses.add(responses.POST, QWEN_FULL_URL, json=qwen_response, status=200)

        gateway = Gateway(default_provider="qwen", default_api_key=mock_api_key)
        request = ChatCompletionRequest(model="qwen-plus", messages=sample_messages)
        result = gateway.chat_complete(request)

        assert result.provider == "qwen"
        assert result.choices[0].message.content == "Hello! How can I assist you today?"

    @responses.activate
    def test_passes_request_params_through(self, mock_api_key, sample_messages, openai_response):
        """Test Gateway passes request params to chat_complete."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        gateway = Gateway(
            default_provider="openai",
            default_api_key=mock_api_key,
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=sample_messages,
            temperature=0.7,
            max_tokens=100,
        )
        gateway.chat_complete(request)

        import json

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["temperature"] == 0.7
        assert request_body["max_tokens"] == 100


class TestGatewayIntegration:
    """Integration tests for Gateway."""

    @responses.activate
    def test_multiple_providers_same_gateway(
        self,
        mock_api_key,
        mocker,
        sample_messages,
        openai_response,
        anthropic_response,
        google_response,
        grok_response,
        deepseek_response,
        qwen_response,
        llama_response,
    ):
        """One Gateway instance routes correctly to every supported provider.

        The seven providers split into two transport classes, so the test
        combines both mocking strategies:
          - SDK-based (httpx under the hood) — OpenAI, Grok, DeepSeek;
            mocked by patching the ``OpenAI`` class imported by each module.
          - Direct ``requests``-based — Anthropic, Google, Qwen, Llama;
            mocked with the ``responses`` library.
        """
        # SDK-based providers: patch the OpenAI class in each provider module.
        openai_mock = _patch_openai_class_in(
            "rezunate_llm_sdk.providers.openai_provider", mocker, openai_response
        )
        grok_mock = _patch_openai_class_in(
            "rezunate_llm_sdk.providers.grok_provider", mocker, grok_response
        )
        deepseek_mock = _patch_openai_class_in(
            "rezunate_llm_sdk.providers.deepseek_provider", mocker, deepseek_response
        )

        # HTTP-based providers: register URL stubs with responses.
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json=anthropic_response,
            status=200,
        )
        responses.add(
            responses.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=google_response,
            status=200,
        )
        responses.add(responses.POST, QWEN_FULL_URL, json=qwen_response, status=200)
        responses.add(responses.POST, LLAMA_FULL_URL, json=llama_response, status=200)

        gateway = Gateway()

        result_openai = gateway.chat_complete(
            ChatCompletionRequest(model="gpt-4", messages=sample_messages),
            provider="openai",
            api_key=mock_api_key,
        )
        result_anthropic = gateway.chat_complete(
            ChatCompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=sample_messages,
                max_tokens=100,
            ),
            provider="anthropic",
            api_key=mock_api_key,
        )
        result_google = gateway.chat_complete(
            ChatCompletionRequest(model="gemini-2.0-flash", messages=sample_messages),
            provider="google",
            api_key=mock_api_key,
        )
        result_grok = gateway.chat_complete(
            ChatCompletionRequest(model="grok-3-mini", messages=sample_messages),
            provider="grok",
            api_key=mock_api_key,
        )
        result_deepseek = gateway.chat_complete(
            ChatCompletionRequest(model="deepseek-chat", messages=sample_messages),
            provider="deepseek",
            api_key=mock_api_key,
        )
        result_qwen = gateway.chat_complete(
            ChatCompletionRequest(model="qwen-plus", messages=sample_messages),
            provider="qwen",
            api_key=mock_api_key,
        )
        result_llama = gateway.chat_complete(
            ChatCompletionRequest(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=sample_messages,
            ),
            provider="llama",
            api_key=mock_api_key,
        )

        # Each call was routed to the right provider.
        assert result_openai.provider == "openai"
        assert result_anthropic.provider == "anthropic"
        assert result_google.provider == "google"
        assert result_grok.provider == "grok"
        assert result_deepseek.provider == "deepseek"
        assert result_qwen.provider == "qwen"
        assert result_llama.provider == "llama"

        # SDK-based providers each invoked once via the patched OpenAI client.
        openai_mock.assert_called_once()
        grok_mock.assert_called_once()
        deepseek_mock.assert_called_once()
        # responses.calls captures the four HTTP-mocked providers.
        assert len(responses.calls) == 4
