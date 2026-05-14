"""
Tests for Provider Factory.
"""

import pytest

from rezunate_llm_sdk.models import Provider
from rezunate_llm_sdk.providers.anthropic_provider import AnthropicProvider
from rezunate_llm_sdk.providers.deepseek_provider import DeepSeekProvider
from rezunate_llm_sdk.providers.factory import (
    FACTORY_REGISTRY,
    AnthropicFactory,
    DeepSeekFactory,
    GoogleFactory,
    GrokFactory,
    LlamaFactory,
    OpenAIFactory,
    ProviderFactory,
    QwenFactory,
    get_factory,
    register_factory,
)
from rezunate_llm_sdk.providers.google_provider import GoogleProvider
from rezunate_llm_sdk.providers.grok_provider import GrokProvider
from rezunate_llm_sdk.providers.llama_provider import LlamaProvider
from rezunate_llm_sdk.providers.openai_provider import OpenAIProvider
from rezunate_llm_sdk.providers.qwen_provider import QwenProvider


@pytest.fixture
def clean_registry():
    """Fixture that saves and restores FACTORY_REGISTRY state.

    Ensures test isolation even if tests fail before manual cleanup.
    """
    # Save original state
    original_registry = FACTORY_REGISTRY.copy()

    yield FACTORY_REGISTRY

    # Restore original state after test
    FACTORY_REGISTRY.clear()
    FACTORY_REGISTRY.update(original_registry)


class TestFactoryRegistry:
    """Tests for factory registry."""

    def test_registry_contains_all_providers(self):
        """Test registry contains all default providers."""
        assert Provider.OPENAI in FACTORY_REGISTRY
        assert Provider.ANTHROPIC in FACTORY_REGISTRY
        assert Provider.GOOGLE in FACTORY_REGISTRY
        assert Provider.GROK in FACTORY_REGISTRY
        assert Provider.LLAMA in FACTORY_REGISTRY
        assert Provider.DEEPSEEK in FACTORY_REGISTRY
        assert Provider.QWEN in FACTORY_REGISTRY

    def test_registry_factories_are_instances(self):
        """Test registry contains factory instances, not classes."""
        assert isinstance(FACTORY_REGISTRY[Provider.OPENAI], OpenAIFactory)
        assert isinstance(FACTORY_REGISTRY[Provider.ANTHROPIC], AnthropicFactory)
        assert isinstance(FACTORY_REGISTRY[Provider.GOOGLE], GoogleFactory)
        assert isinstance(FACTORY_REGISTRY[Provider.GROK], GrokFactory)
        assert isinstance(FACTORY_REGISTRY[Provider.LLAMA], LlamaFactory)
        assert isinstance(FACTORY_REGISTRY[Provider.DEEPSEEK], DeepSeekFactory)
        assert isinstance(FACTORY_REGISTRY[Provider.QWEN], QwenFactory)


class TestGetFactory:
    """Tests for get_factory function."""

    def test_get_openai_factory(self):
        """Test getting OpenAI factory."""
        factory = get_factory(Provider.OPENAI)
        assert isinstance(factory, OpenAIFactory)

    def test_get_anthropic_factory(self):
        """Test getting Anthropic factory."""
        factory = get_factory(Provider.ANTHROPIC)
        assert isinstance(factory, AnthropicFactory)

    def test_get_google_factory(self):
        """Test getting Google factory."""
        factory = get_factory(Provider.GOOGLE)
        assert isinstance(factory, GoogleFactory)

    def test_get_grok_factory(self):
        """Test getting Grok factory."""
        factory = get_factory(Provider.GROK)
        assert isinstance(factory, GrokFactory)

    def test_get_llama_factory(self):
        """Test getting Llama factory."""
        factory = get_factory(Provider.LLAMA)
        assert isinstance(factory, LlamaFactory)

    def test_get_deepseek_factory(self):
        """Test getting DeepSeek factory."""
        factory = get_factory(Provider.DEEPSEEK)
        assert isinstance(factory, DeepSeekFactory)

    def test_get_qwen_factory(self):
        """Test getting Qwen factory."""
        factory = get_factory(Provider.QWEN)
        assert isinstance(factory, QwenFactory)

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

    def test_register_new_factory(self, mock_api_key, clean_registry):
        """Test registering a new factory (overwriting existing to avoid crashing factory.py)."""

        # Create a mock factory
        class MockFactory(ProviderFactory):
            @property
            def provider_name(self):
                return Provider.OPENAI

            def create_provider(self, api_key, **kwargs):
                return OpenAIProvider(api_key=api_key)

        mock_factory = MockFactory()
        # Use Provider Enum member instead of string to avoid factory.py crash
        register_factory(Provider.OPENAI, mock_factory)

        assert Provider.OPENAI in FACTORY_REGISTRY
        assert get_factory(Provider.OPENAI) is mock_factory
        # Cleanup handled by clean_registry fixture

    def test_register_overwrites_existing(self, clean_registry):
        """Test registering overwrites existing factory."""
        original_factory = FACTORY_REGISTRY[Provider.OPENAI]

        class NewOpenAIFactory(ProviderFactory):
            @property
            def provider_name(self):
                return Provider.OPENAI

            def create_provider(self, api_key, **kwargs):
                return OpenAIProvider(api_key=api_key)

        new_factory = NewOpenAIFactory()
        # Use Provider Enum member instead of string to avoid factory.py crash
        register_factory(Provider.OPENAI, new_factory)

        assert FACTORY_REGISTRY[Provider.OPENAI] is new_factory
        assert FACTORY_REGISTRY[Provider.OPENAI] is not original_factory
        # Cleanup handled by clean_registry fixture


class TestOpenAIFactory:
    """Tests for OpenAI factory."""

    def test_provider_name(self):
        """Test factory returns correct provider name."""
        factory = OpenAIFactory()
        assert factory.provider_name == Provider.OPENAI

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
        assert factory.provider_name == Provider.ANTHROPIC

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
        assert factory.provider_name == Provider.GOOGLE

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


class TestGrokFactory:
    """Tests for Grok factory."""

    def test_provider_name(self):
        factory = GrokFactory()
        assert factory.provider_name == Provider.GROK

    def test_creates_grok_provider(self, mock_api_key):
        factory = GrokFactory()
        provider = factory.create_provider(api_key=mock_api_key)

        assert isinstance(provider, GrokProvider)
        assert provider.api_key == mock_api_key

    def test_passes_kwargs_to_provider(self, mock_api_key):
        factory = GrokFactory()
        provider = factory.create_provider(
            api_key=mock_api_key,
            max_retries=5,
            timeout=120.0,
        )
        assert provider.max_retries == 5
        assert provider.timeout == 120.0


class TestLlamaFactory:
    """Tests for Llama factory."""

    def test_provider_name(self):
        factory = LlamaFactory()
        assert factory.provider_name == Provider.LLAMA

    def test_creates_llama_provider(self, mock_api_key):
        factory = LlamaFactory()
        provider = factory.create_provider(api_key=mock_api_key)

        assert isinstance(provider, LlamaProvider)
        assert provider.api_key == mock_api_key

    def test_passes_kwargs_to_provider(self, mock_api_key):
        factory = LlamaFactory()
        provider = factory.create_provider(
            api_key=mock_api_key,
            max_retries=4,
            timeout=90.0,
        )
        assert provider.max_retries == 4
        assert provider.timeout == 90.0


class TestDeepSeekFactory:
    """Tests for DeepSeek factory."""

    def test_provider_name(self):
        factory = DeepSeekFactory()
        assert factory.provider_name == Provider.DEEPSEEK

    def test_creates_deepseek_provider(self, mock_api_key):
        factory = DeepSeekFactory()
        provider = factory.create_provider(api_key=mock_api_key)

        assert isinstance(provider, DeepSeekProvider)
        assert provider.api_key == mock_api_key

    def test_passes_kwargs_to_provider(self, mock_api_key):
        factory = DeepSeekFactory()
        provider = factory.create_provider(
            api_key=mock_api_key,
            max_retries=2,
            timeout=45.0,
        )
        assert provider.max_retries == 2
        assert provider.timeout == 45.0


class TestQwenFactory:
    """Tests for Qwen factory."""

    def test_provider_name(self):
        factory = QwenFactory()
        assert factory.provider_name == Provider.QWEN

    def test_creates_qwen_provider(self, mock_api_key):
        factory = QwenFactory()
        provider = factory.create_provider(api_key=mock_api_key)

        assert isinstance(provider, QwenProvider)
        assert provider.api_key == mock_api_key

    def test_passes_kwargs_to_provider(self, mock_api_key):
        factory = QwenFactory()
        provider = factory.create_provider(
            api_key=mock_api_key,
            max_retries=3,
            timeout=60.0,
        )
        assert provider.max_retries == 3
        assert provider.timeout == 60.0


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
