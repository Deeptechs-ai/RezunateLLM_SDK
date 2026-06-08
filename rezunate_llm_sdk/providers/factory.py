"""
Provider Factory - Factory Method Pattern.
Each provider has its own factory class for creating instances.
"""

from abc import ABC, abstractmethod

from rezunate_llm_sdk.models import Provider
from rezunate_llm_sdk.providers.base import BaseProvider


class ProviderFactory(ABC):
    """
    Abstract Factory class for creating provider instances.
    Each concrete factory must implement the create_provider method.
    """

    @abstractmethod
    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        """
        Create and return a provider instance.

        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific arguments

        Returns:
            BaseProvider instance
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> Provider:
        """Return the name of the provider this factory creates."""
        pass


class OpenAIFactory(ProviderFactory):
    """Factory for creating OpenAI provider instances."""

    @property
    def provider_name(self) -> Provider:
        return Provider.OPENAI

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from rezunate_llm_sdk.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=api_key, **kwargs)


class AnthropicFactory(ProviderFactory):
    """Factory for creating Anthropic provider instances."""

    @property
    def provider_name(self) -> Provider:
        return Provider.ANTHROPIC

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from rezunate_llm_sdk.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, **kwargs)


class GoogleFactory(ProviderFactory):
    """Factory for creating Google/Gemini provider instances."""

    @property
    def provider_name(self) -> Provider:
        return Provider.GOOGLE

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from rezunate_llm_sdk.providers.google_provider import GoogleProvider

        return GoogleProvider(api_key=api_key, **kwargs)


class GrokFactory(ProviderFactory):
    """Factory for creating Grok (xAI) provider instances."""

    @property
    def provider_name(self) -> Provider:
        return Provider.GROK

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from rezunate_llm_sdk.providers.grok_provider import GrokProvider

        return GrokProvider(api_key=api_key, **kwargs)


class LlamaFactory(ProviderFactory):
    """Factory for creating Llama (Meta) provider instances."""

    @property
    def provider_name(self) -> Provider:
        return Provider.LLAMA

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from rezunate_llm_sdk.providers.llama_provider import LlamaProvider

        return LlamaProvider(api_key=api_key, **kwargs)


class DeepSeekFactory(ProviderFactory):
    """Factory for creating DeepSeek provider instances."""

    @property
    def provider_name(self) -> Provider:
        return Provider.DEEPSEEK

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from rezunate_llm_sdk.providers.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(api_key=api_key, **kwargs)


class QwenFactory(ProviderFactory):
    """Factory for creating Qwen (Alibaba) provider instances."""

    @property
    def provider_name(self) -> Provider:
        return Provider.QWEN

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from rezunate_llm_sdk.providers.qwen_provider import QwenProvider

        return QwenProvider(api_key=api_key, **kwargs)


# Factory Registry - maps provider names to factory instances
FACTORY_REGISTRY: dict[Provider, ProviderFactory] = {
    Provider.OPENAI: OpenAIFactory(),
    Provider.ANTHROPIC: AnthropicFactory(),
    Provider.GOOGLE: GoogleFactory(),
    Provider.GROK: GrokFactory(),
    Provider.LLAMA: LlamaFactory(),
    Provider.DEEPSEEK: DeepSeekFactory(),
    Provider.QWEN: QwenFactory(),
}


def get_factory(provider_name: str | Provider) -> ProviderFactory:
    """
    Get a factory instance by provider name.

    Args:
        provider_name: Name of the provider (openai, anthropic, google)

    Returns:
        ProviderFactory instance

    Raises:
        ValueError: If provider is not found
    """
    if isinstance(provider_name, str):
        try:
            provider_name = Provider(provider_name.lower())
        except ValueError:
            raise ValueError(
                f"Unknown provider: '{provider_name}'. "
                f"Available providers: {', '.join(p.value for p in Provider)}"
            ) from None

    return FACTORY_REGISTRY[provider_name]


def register_factory(name: str, factory: ProviderFactory) -> None:
    """
    Register a new factory in the registry.

    Args:
        name: Provider name to register
        factory: Factory instance to register
    """
    FACTORY_REGISTRY[name] = factory
