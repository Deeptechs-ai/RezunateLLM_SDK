"""LLM Router SDK.

A unified Python library for chat completions across multiple AI providers
(OpenAI, Anthropic, Google) using OpenAI-compatible request/response format.
"""

from rezunate_llm_sdk.client import RouterAPIError, RouterClient
from rezunate_llm_sdk.gateway import Gateway, chat_complete, get_available_providers
from rezunate_llm_sdk.guardrails import (
    GuardrailsError,
    check_guardrails,
    load_guardrails,
)
from rezunate_llm_sdk.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceChunk,
    ChoiceDelta,
    DetectedEntity,
    FunctionCall,
    FunctionDefinition,
    GuardrailRule,
    GuardrailsConfig,
    Message,
    PromptResponse,
    ScanResponse,
    Tool,
    ToolCall,
    ToolChoiceFunction,
    ToolChoiceOption,
)
from rezunate_llm_sdk.prompts import render_prompt
from rezunate_llm_sdk.providers import (
    AnthropicProvider,
    BaseProvider,
    DeepSeekProvider,
    GoogleProvider,
    GrokProvider,
    LlamaProvider,
    OpenAIProvider,
    QwenProvider,
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
    "ChatCompletionChunk",
    "ChoiceChunk",
    "ChoiceDelta",
    "DetectedEntity",
    "FunctionCall",
    "FunctionDefinition",
    "Message",
    "PromptResponse",
    "ScanResponse",
    "Tool",
    "ToolCall",
    "ToolChoiceFunction",
    "ToolChoiceOption",
    "render_prompt",
    "AnthropicProvider",
    "BaseProvider",
    "DeepSeekProvider",
    "GoogleProvider",
    "GrokProvider",
    "LlamaProvider",
    "OpenAIProvider",
    "QwenProvider",
    "get_provider",
    "list_providers",
]

__version__ = "0.1.0"
