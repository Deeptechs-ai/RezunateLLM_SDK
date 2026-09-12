"""
Provider Registry.

Uses Factory Method Pattern for creating provider instances.
"""

from rezunate_llm_sdk.models import Provider
from rezunate_llm_sdk.providers.anthropic_provider import AnthropicProvider
from rezunate_llm_sdk.providers.base import BaseProvider
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

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "DeepSeekProvider",
    "GoogleProvider",
    "GrokProvider",
    "LlamaProvider",
    "OpenAIProvider",
    "QwenProvider",
    "FACTORY_REGISTRY",
    "AnthropicFactory",
    "DeepSeekFactory",
    "GoogleFactory",
    "GrokFactory",
    "LlamaFactory",
    "OpenAIFactory",
    "ProviderFactory",
    "QwenFactory",
    "get_factory",
    "register_factory",
    "get_provider",
    "list_providers",
]


def get_provider(provider_name: str | Provider, api_key: str, **kwargs) -> BaseProvider:
    """
    Get a provider instance by name using Factory Method Pattern.

    Args:
        provider_name: Name of the provider (any Provider value, e.g. openai,
            anthropic, google, grok, llama, deepseek, qwen)
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
