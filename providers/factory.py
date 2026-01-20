"""
Provider Factory - Factory Method Pattern.
Each provider has its own factory class for creating instances.
"""

from abc import ABC, abstractmethod

from providers.base import BaseProvider


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
    def provider_name(self) -> str:
        """Return the name of the provider this factory creates."""
        pass


class OpenAIFactory(ProviderFactory):
    """Factory for creating OpenAI provider instances."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=api_key, **kwargs)


class AnthropicFactory(ProviderFactory):
    """Factory for creating Anthropic provider instances."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, **kwargs)


class GoogleFactory(ProviderFactory):
    """Factory for creating Google/Gemini provider instances."""

    @property
    def provider_name(self) -> str:
        return "google"

    def create_provider(self, api_key: str, **kwargs) -> BaseProvider:
        from providers.google_provider import GoogleProvider

        return GoogleProvider(api_key=api_key, **kwargs)


# Factory Registry - maps provider names to factory instances
FACTORY_REGISTRY: dict[str, ProviderFactory] = {
    "openai": OpenAIFactory(),
    "anthropic": AnthropicFactory(),
    "google": GoogleFactory(),
}


def get_factory(provider_name: str) -> ProviderFactory:
    """
    Get a factory instance by provider name.

    Args:
        provider_name: Name of the provider (openai, anthropic, google)

    Returns:
        ProviderFactory instance

    Raises:
        ValueError: If provider is not found
    """
    if provider_name not in FACTORY_REGISTRY:
        available = ", ".join(FACTORY_REGISTRY.keys())
        raise ValueError(f"Unknown provider: '{provider_name}'. Available providers: {available}")
    return FACTORY_REGISTRY[provider_name]


def register_factory(name: str, factory: ProviderFactory) -> None:
    """
    Register a new factory in the registry.

    Args:
        name: Provider name to register
        factory: Factory instance to register
    """
    FACTORY_REGISTRY[name] = factory
