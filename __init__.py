"""LLM Router SDK.

A unified Python library for chat completions across multiple AI providers
(OpenAI, Anthropic, Google) using OpenAI-compatible request/response format.
"""

from gateway import Gateway, chat_complete, get_available_providers
from models import ChatCompletionRequest, ChatCompletionResponse, Message
from providers import (
    PROVIDERS,
    AnthropicProvider,
    BaseProvider,
    GoogleProvider,
    OpenAIProvider,
    get_provider,
    list_providers,
)

__all__ = [
    "Gateway",
    "chat_complete",
    "get_available_providers",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Message",
    "PROVIDERS",
    "AnthropicProvider",
    "BaseProvider",
    "GoogleProvider",
    "OpenAIProvider",
    "get_provider",
    "list_providers",
]

__version__ = "0.1.0"
