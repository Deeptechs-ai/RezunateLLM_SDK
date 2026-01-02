"""
Provider Registry.
Uses Factory Method Pattern for creating provider instances.
"""

from typing import Dict, Type
from providers.base import BaseProvider
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.google_provider import GoogleProvider
from providers.factory import (
    ProviderFactory,
    OpenAIFactory,
    AnthropicFactory,
    GoogleFactory,
    FACTORY_REGISTRY,
    get_factory,
    register_factory,
)


# ============== Provider Registry ==============
# Keep for backward compatibility and direct access
PROVIDERS: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def get_provider(provider_name: str, api_key: str, **kwargs) -> BaseProvider:
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


def list_providers() -> list:
    """Return list of available provider names."""
    return list(FACTORY_REGISTRY.keys())


# Export
__all__ = [
    # Provider classes
    "PROVIDERS",
    "get_provider",
    "list_providers",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    # Factory classes
    "ProviderFactory",
    "OpenAIFactory",
    "AnthropicFactory",
    "GoogleFactory",
    "FACTORY_REGISTRY",
    "get_factory",
    "register_factory",
]
