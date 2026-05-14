"""
Llama (Meta) native API Pydantic models.

These represent Meta's native Llama API request/response format
(NOT the /compat/v1 OpenAI-compatible endpoint). Used for type safety
and validation in the native Llama provider.

Key differences from OpenAI:
  - Response uses ``completion_message`` (single object) instead of
    ``choices`` (array).
  - Message content is a typed block ``{type: "text", text: "..."}``,
    not a plain string.
  - Token counts arrive in a ``metrics`` array, not a ``usage`` object.

Reference: Meta Llama API documentation at llama.developer.meta.com.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Request Models


class LlamaMessage(BaseModel):
    """Message in Meta Llama native format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LlamaRequest(BaseModel):
    """Meta Llama native chat completion request."""

    model: str
    messages: list[LlamaMessage]
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    stream: bool | None = None
    tools: list | None = None
    response_format: dict | None = None

    model_config = {"extra": "allow"}


# Response Models


class LlamaContentBlock(BaseModel):
    """Content block in a Meta Llama response message."""

    type: Literal["text"] = "text"
    text: str = ""


class LlamaCompletionMessage(BaseModel):
    """The single completion message in a Meta Llama response."""

    role: Literal["assistant"] = "assistant"
    content: LlamaContentBlock = Field(default_factory=LlamaContentBlock)
    stop_reason: str | None = None
    tool_calls: list | None = None


class LlamaMetric(BaseModel):
    """A single metric entry in Meta Llama's metrics array."""

    metric: str
    value: float
    unit: str | None = None


class LlamaResponse(BaseModel):
    """Meta Llama native chat completion response."""

    id: str | None = None
    completion_message: LlamaCompletionMessage = Field(default_factory=LlamaCompletionMessage)
    metrics: list[LlamaMetric] = Field(default_factory=list)

    model_config = {"extra": "allow"}
