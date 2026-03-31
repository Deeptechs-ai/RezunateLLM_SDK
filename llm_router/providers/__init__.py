"""
Provider Registry.

Uses Factory Method Pattern for creating provider instances.
"""

from llm_router.models import Provider
from llm_router.providers.anthropic_provider import AnthropicProvider
from llm_router.providers.base import BaseProvider
from llm_router.providers.factory import (
    FACTORY_REGISTRY,
    AnthropicFactory,
    GoogleFactory,
    OpenAIFactory,
    ProviderFactory,
    get_factory,
    register_factory,
)
from llm_router.providers.google_provider import GoogleProvider
from llm_router.providers.openai_provider import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "GoogleProvider",
    "OpenAIProvider",
    "FACTORY_REGISTRY",
    "AnthropicFactory",
    "GoogleFactory",
    "OpenAIFactory",
    "ProviderFactory",
    "get_factory",
    "register_factory",
    "get_provider",
    "list_providers",
]


def get_provider(provider_name: str | Provider, api_key: str, **kwargs) -> BaseProvider:
    """
    Get a provider instance by name using Factory Method Pattern.

    Args:
        provider_name: Name of the provider (openai, anthropic, google)
        api_key: API key for the provider
        **kwargs: Additional provider-specific arguments

    Returns:
        Provider instance

    Raises:
        ValueError: If provider is not found
    """
    factory = get_factory(provider_name)
    return factory.create_provider(api_key=api_key, **kwargs)


def list_providers() -> list[Provider]:
    """Return list of available provider names."""
    return list(FACTORY_REGISTRY.keys())
