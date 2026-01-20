"""LLM Router SDK.

A unified Python library for chat completions across multiple AI providers
(OpenAI, Anthropic, Google) using OpenAI-compatible request/response format.
"""

from gateway import Gateway, chat_complete, get_available_providers
from providers import (
    PROVIDERS,
    AnthropicProvider,
    BaseProvider,
    GoogleProvider,
    OpenAIProvider,
    get_provider,
    list_providers,
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    "Gateway",
    "chat_complete",
    "get_available_providers",
    # Provider utilities
    "PROVIDERS",
    "get_provider",
    "list_providers",
    # Provider classes
    "AnthropicProvider",
    "BaseProvider",
    "GoogleProvider",
    "OpenAIProvider",
]
