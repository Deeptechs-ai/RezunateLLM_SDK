"""LLM Router SDK.

A unified Python library for chat completions across multiple AI providers
(OpenAI, Anthropic, Google) using OpenAI-compatible request/response format.
"""

from llm_router.gateway import Gateway, chat_complete, get_available_providers
from llm_router.models import ChatCompletionRequest, ChatCompletionResponse, Message
from llm_router.providers import (
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
