"""
Tests for Provider Factory.
"""

import pytest
from providers.factory import (
    ProviderFactory,
    OpenAIFactory,
    AnthropicFactory,
    GoogleFactory,
    FACTORY_REGISTRY,
    get_factory,
    register_factory,
)
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.google_provider import GoogleProvider
from providers.base import BaseProvider


class TestFactoryRegistry:
    """Tests for factory registry."""

    def test_registry_contains_all_providers(self):
        """Test registry contains all default providers."""
        assert "openai" in FACTORY_REGISTRY
        assert "anthropic" in FACTORY_REGISTRY
        assert "google" in FACTORY_REGISTRY

    def test_registry_factories_are_instances(self):
        """Test registry contains factory instances, not classes."""
        assert isinstance(FACTORY_REGISTRY["openai"], OpenAIFactory)
        assert isinstance(FACTORY_REGISTRY["anthropic"], AnthropicFactory)
        assert isinstance(FACTORY_REGISTRY["google"], GoogleFactory)


class TestGetFactory:
    """Tests for get_factory function."""

    def test_get_openai_factory(self):
        """Test getting OpenAI factory."""
        factory = get_factory("openai")
        assert isinstance(factory, OpenAIFactory)

    def test_get_anthropic_factory(self):
        """Test getting Anthropic factory."""
        factory = get_factory("anthropic")
        assert isinstance(factory, AnthropicFactory)

    def test_get_google_factory(self):
        """Test getting Google factory."""
        factory = get_factory("google")
        assert isinstance(factory, GoogleFactory)

    def test_unknown_provider_raises_error(self):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_factory("unknown_provider")

        assert "Unknown provider: 'unknown_provider'" in str(exc_info.value)
        assert "Available providers:" in str(exc_info.value)

    def test_error_message_lists_available_providers(self):
        """Test error message includes available providers."""
        with pytest.raises(ValueError) as exc_info:
            get_factory("invalid")

        error_msg = str(exc_info.value)
        assert "openai" in error_msg
        assert "anthropic" in error_msg
        assert "google" in error_msg


class TestRegisterFactory:
    """Tests for register_factory function."""

    def test_register_new_factory(self, mock_api_key):
        """Test registering a new factory."""
        # Create a mock factory
        class MockFactory(ProviderFactory):
            @property
            def provider_name(self):
                return "mock"

            def create_provider(self, api_key, **kwargs):
                return OpenAIProvider(api_key=api_key)  # Return any provider for test

        mock_factory = MockFactory()
        register_factory("mock_provider", mock_factory)

        assert "mock_provider" in FACTORY_REGISTRY
        assert get_factory("mock_provider") is mock_factory

        # Cleanup
        del FACTORY_REGISTRY["mock_provider"]

    def test_register_overwrites_existing(self):
        """Test registering overwrites existing factory."""
        original_factory = FACTORY_REGISTRY["openai"]

        class NewOpenAIFactory(ProviderFactory):
            @property
            def provider_name(self):
                return "openai"

            def create_provider(self, api_key, **kwargs):
                return OpenAIProvider(api_key=api_key)

        new_factory = NewOpenAIFactory()
        register_factory("openai", new_factory)

        assert FACTORY_REGISTRY["openai"] is new_factory

        # Restore original
        FACTORY_REGISTRY["openai"] = original_factory


class TestOpenAIFactory:
    """Tests for OpenAI factory."""

    def test_provider_name(self):
        """Test factory returns correct provider name."""
        factory = OpenAIFactory()
        assert factory.provider_name == "openai"

    def test_creates_openai_provider(self, mock_api_key):
        """Test factory creates OpenAI provider."""
        factory = OpenAIFactory()
        provider = factory.create_provider(api_key=mock_api_key)

        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == mock_api_key

    def test_passes_kwargs_to_provider(self, mock_api_key):
        """Test factory passes kwargs to provider."""
        factory = OpenAIFactory()
        provider = factory.create_provider(
            api_key=mock_api_key,
            max_retries=5,
            timeout=120.0,
        )

        assert provider.max_retries == 5
        assert provider.timeout == 120.0


class TestAnthropicFactory:
    """Tests for Anthropic factory."""

    def test_provider_name(self):
        """Test factory returns correct provider name."""
        factory = AnthropicFactory()
        assert factory.provider_name == "anthropic"

    def test_creates_anthropic_provider(self, mock_api_key):
        """Test factory creates Anthropic provider."""
        factory = AnthropicFactory()
        provider = factory.create_provider(api_key=mock_api_key)

        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == mock_api_key

    def test_passes_kwargs_to_provider(self, mock_api_key):
        """Test factory passes kwargs to provider."""
        factory = AnthropicFactory()
        provider = factory.create_provider(
            api_key=mock_api_key,
            max_retries=10,
            retry_delay=2.0,
        )

        assert provider.max_retries == 10
        assert provider.retry_delay == 2.0


class TestGoogleFactory:
    """Tests for Google factory."""

    def test_provider_name(self):
        """Test factory returns correct provider name."""
        factory = GoogleFactory()
        assert factory.provider_name == "google"

    def test_creates_google_provider(self, mock_api_key):
        """Test factory creates Google provider."""
        factory = GoogleFactory()
        provider = factory.create_provider(api_key=mock_api_key)

        assert isinstance(provider, GoogleProvider)
        assert provider.api_key == mock_api_key

    def test_passes_kwargs_to_provider(self, mock_api_key):
        """Test factory passes kwargs to Google provider."""
        factory = GoogleFactory()
        provider = factory.create_provider(
            api_key=mock_api_key,
            timeout=120.0,
        )

        assert provider.timeout == 120.0


class TestProviderFactoryABC:
    """Tests for ProviderFactory abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test ProviderFactory cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProviderFactory()

    def test_must_implement_create_provider(self):
        """Test subclass must implement create_provider."""
        class IncompleteFactory(ProviderFactory):
            @property
            def provider_name(self):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteFactory()

    def test_must_implement_provider_name(self):
        """Test subclass must implement provider_name."""
        class IncompleteFactory(ProviderFactory):
            def create_provider(self, api_key, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteFactory()
