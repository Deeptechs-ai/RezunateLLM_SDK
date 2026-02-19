"""LLM Router SDK.

A unified Python library for chat completions across multiple AI providers
(OpenAI, Anthropic, Google) using OpenAI-compatible request/response format.
"""

from llm_router.client import RouterAPIError, RouterClient
from llm_router.gateway import Gateway, chat_complete, get_available_providers
from llm_router.guardrails import GuardrailsError, check_guardrails, load_guardrails
from llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    GuardrailRule,
    GuardrailsConfig,
    Message,
    PromptResponse,
)
from llm_router.prompts import render_prompt
from llm_router.providers import (
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
    "RouterClient",
    "RouterAPIError",
    "GuardrailsError",
    "GuardrailsConfig",
    "GuardrailRule",
    "load_guardrails",
    "check_guardrails",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Message",
    "PromptResponse",
    "render_prompt",
    "AnthropicProvider",
    "BaseProvider",
    "GoogleProvider",
    "OpenAIProvider",
    "get_provider",
    "list_providers",
]

__version__ = "0.1.0"
