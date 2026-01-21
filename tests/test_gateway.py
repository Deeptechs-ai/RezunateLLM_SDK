"""
Tests for Gateway (main router).
"""

import pytest
import responses

from gateway import Gateway, chat_complete, get_available_providers


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

        result = chat_complete(
            provider="openai",
            api_key=mock_api_key,
            model="gpt-4",
            messages=sample_messages,
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

        result = chat_complete(
            provider="anthropic",
            api_key=mock_api_key,
            model="claude-sonnet-4-20250514",
            messages=sample_messages,
            max_tokens=100,
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

        result = chat_complete(
            provider="google",
            api_key=mock_api_key,
            model="gemini-2.0-flash",
            messages=sample_messages,
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

        chat_complete(
            provider="openai",
            api_key=mock_api_key,
            model="gpt-4",
            messages=sample_messages,
            temperature=0.5,
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

        chat_complete(
            provider="openai",
            api_key=mock_api_key,
            model="gpt-4",
            messages=sample_messages,
            max_tokens=100,
        )

        import json

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["max_tokens"] == 100

    @responses.activate
    def test_passes_kwargs(self, mock_api_key, sample_messages, openai_response):
        """Test additional kwargs are passed to provider."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        chat_complete(
            provider="openai",
            api_key=mock_api_key,
            model="gpt-4",
            messages=sample_messages,
            top_p=0.9,
            frequency_penalty=0.5,
        )

        import json

        request_body = json.loads(responses.calls[0].request.body)
        assert request_body["top_p"] == 0.9
        assert request_body["frequency_penalty"] == 0.5

    def test_unknown_provider_raises_error(self, mock_api_key, sample_messages):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            chat_complete(
                provider="unknown",
                api_key=mock_api_key,
                model="some-model",
                messages=sample_messages,
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

        result = chat_complete(
            provider="anthropic",
            api_key=mock_api_key,
            model="claude-sonnet-4-20250514",
            messages=sample_messages,
            max_tokens=100,
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

        result = gateway.chat_complete(
            messages=sample_messages,
            model="gpt-4",
        )

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

        result = gateway.chat_complete(
            messages=sample_messages,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            api_key=mock_api_key,
            max_tokens=100,
        )

        assert result.provider == "anthropic"

    def test_chat_complete_requires_provider(self, mock_api_key, sample_messages):
        """Test chat_complete raises error if no provider."""
        gateway = Gateway(default_api_key=mock_api_key)

        with pytest.raises(ValueError) as exc_info:
            gateway.chat_complete(
                messages=sample_messages,
                model="gpt-4",
            )

        assert "Provider must be specified" in str(exc_info.value)

    def test_chat_complete_requires_api_key(self, sample_messages):
        """Test chat_complete raises error if no api_key."""
        gateway = Gateway(default_provider="openai")

        with pytest.raises(ValueError) as exc_info:
            gateway.chat_complete(
                messages=sample_messages,
                model="gpt-4",
            )

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
    def test_passes_kwargs_through(self, mock_api_key, sample_messages, openai_response):
        """Test Gateway passes kwargs to chat_complete."""
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

        gateway.chat_complete(
            messages=sample_messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
        )

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
        result1 = gateway.chat_complete(
            provider="openai",
            api_key=mock_api_key,
            model="gpt-4",
            messages=sample_messages,
        )

        # Call Anthropic
        result2 = gateway.chat_complete(
            provider="anthropic",
            api_key=mock_api_key,
            model="claude-sonnet-4-20250514",
            messages=sample_messages,
            max_tokens=100,
        )

        assert result1.provider == "openai"
        assert result2.provider == "anthropic"
        assert len(responses.calls) == 2
