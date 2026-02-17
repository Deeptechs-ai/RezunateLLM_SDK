"""
Pydantic models for LLM Router.

Defines request and response models following OpenAI format as the universal standard.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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


class Message(BaseModel):
    """A chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


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

    model_config = {"extra": "allow"}


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseMessage(BaseModel):
    """Message in a chat completion response."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None


class Choice(BaseModel):
    """A completion choice."""

    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = None


class ErrorInfo(BaseModel):
    """Error information for failed requests."""

    message: str
    type: str = "api_error"
    code: int | None = None
    retries_attempted: int | None = None


class ChatCompletionResponse(BaseModel):
    """Response model for chat completion in OpenAI format."""

    id: str | None = None
    object: Literal["chat.completion"] = "chat.completion"
    created: int = 0
    model: str
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    provider: str
    error: ErrorInfo | None = None
