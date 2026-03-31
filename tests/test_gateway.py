"""
Tests for Gateway (main router).
"""

import pytest
import responses

from llm_router.gateway import (
    ChatCompletionRequest,
    Gateway,
    chat_complete,
    get_available_providers,
)


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

    def test_returns_at_least_three_providers(self):
        """Test returns at least three providers."""
        providers = get_available_providers()
        assert len(providers) >= 3


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
        self, mock_api_key, sample_messages, openai_response, anthropic_response
    ):
        """Test using multiple providers with same Gateway instance."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json=anthropic_response,
            status=200,
        )

        gateway = Gateway()

        # Call OpenAI
        request1 = ChatCompletionRequest(model="gpt-4", messages=sample_messages)
        result1 = gateway.chat_complete(
            request1,
            provider="openai",
            api_key=mock_api_key,
        )

        # Call Anthropic
        request2 = ChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=sample_messages,
            max_tokens=100,
        )
        result2 = gateway.chat_complete(
            request2,
            provider="anthropic",
            api_key=mock_api_key,
        )

        assert result1.provider == "openai"
        assert result2.provider == "anthropic"
        assert len(responses.calls) == 2
