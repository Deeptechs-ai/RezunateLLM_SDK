"""
Pydantic models for LLM Router.

Defines request and response models following OpenAI format as the universal standard.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Provider(str, Enum):
    """Available LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROK = "grok"
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"


class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class GuardrailDirection(str, Enum):
    """Direction of the guardrail check (input or output)."""

    INPUT = "input"
    OUTPUT = "output"


class GuardrailAction(str, Enum):
    """Action to take when a guardrail is triggered."""

    BLOCK = "block"
    FLAG = "flag"


class GuardrailRule(BaseModel):
    """A single guardrail rule with a regex pattern."""

    name: str
    pattern: str
    description: str = ""
    action: GuardrailAction = GuardrailAction.BLOCK


class GuardrailsConfig(BaseModel):
    """Configuration holding a list of guardrail rules."""

    guardrails: list[GuardrailRule]


class GuardrailViolation(BaseModel):
    """A single guardrail violation."""

    rule_name: str
    rule_description: str
    direction: GuardrailDirection
    action: GuardrailAction
    match: str


class FunctionCall(BaseModel):
    """OpenAI-shaped function-call payload inside a tool call."""

    name: str
    arguments: str = ""


class ToolCall(BaseModel):
    """A tool/function call requested by the assistant.

    Wire format mirrors OpenAI's ``chat.completions.message.tool_calls[*]`` so it
    serializes cleanly to OpenAI and can be translated to Anthropic ``tool_use``
    and Google ``functionCall`` parts inside the provider transformers.
    """

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    """OpenAI-shaped tool function definition (sent on the request side)."""

    name: str
    description: str | None = None
    # Caller-supplied JSON Schema document.
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """OpenAI-shaped tool entry passed in ``ChatCompletionRequest.tools``."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    """Inner ``function`` block of a ``tool_choice`` selecting a specific tool."""

    name: str


class ToolChoiceOption(BaseModel):
    """Structured ``tool_choice`` payload selecting a specific function."""

    type: Literal["function"] = "function"
    function: ToolChoiceFunction


class Message(BaseModel):
    """A chat message."""

    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion."""

    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    stream: bool | None = None
    user: str | None = None
    tools: list[Tool] | None = None
    tool_choice: Literal["auto", "required", "none"] | ToolChoiceOption | None = None

    model_config = {"extra": "allow"}


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class FinishReason(str, Enum):
    """Reason why a completion finished."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


# Centralized mapping for all provider-specific finish reasons
FINISH_REASON_MAP: dict[str, FinishReason] = {
    # OpenAI-style values — also emitted by xAI (Grok), DeepSeek,
    # Qwen native (DashScope, result_format="message"), and Llama native (Meta).
    # All five providers use the same lowercase vocabulary.
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
    "tool_calls": FinishReason.TOOL_CALLS,
    # DeepSeek-specific — returned by deepseek-reasoner under resource pressure
    "insufficient_system_resource": FinishReason.STOP,
    # Google Gemini
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "SAFETY": FinishReason.CONTENT_FILTER,
    "RECITATION": FinishReason.CONTENT_FILTER,
    "BLOCKLIST": FinishReason.CONTENT_FILTER,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
    "SPII": FinishReason.CONTENT_FILTER,
    "OTHER": FinishReason.STOP,
    "MALFORMED_FUNCTION_CALL": FinishReason.STOP,
    # Anthropic
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
}


class ResponseMessage(BaseModel):
    """Message in a chat completion response."""

    role: Role = Role.ASSISTANT
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    """A completion choice."""

    index: int = 0
    message: ResponseMessage
    finish_reason: FinishReason | None = None


class ErrorInfo(BaseModel):
    """Error information for failed requests."""

    message: str
    type: str = "api_error"
    code: int | None = None
    retries_attempted: int | None = None


class ChatCompletionResponse(BaseModel):
    """Response model for chat completion in OpenAI format."""

    id: str | None = None
    object: str = "chat.completion"
    created: int = 0
    model: str | None = None
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    provider: Provider | None = None
    error: ErrorInfo | None = None


class DetectedEntity(BaseModel):
    """A single PII entity detected by the guardrail service."""

    text: str
    label: str
    score: float
    start: int
    end: int


class ScanResponse(BaseModel):
    """Response from a server-side guardrail scan."""

    entities: list[DetectedEntity]
    action: str
    blocked: bool
    text: str


class PromptResponse(BaseModel):
    """Prompt returned by the LLM-Router API."""

    slug_id: str
    name: str
    content: str
    description: str | None = None
    current_version: int
    workspace_id: int
    created_by: int
    input_variables: list[str] | None = None
    created_at: datetime
    updated_at: datetime
