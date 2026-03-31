"""
Anthropic-specific Pydantic models.

These models represent Anthropic's API request/response format.
Used for type safety and validation in the Anthropic provider.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Request Models


class AnthropicMessage(BaseModel):
    """Message in Anthropic format."""

    role: Literal["user", "assistant"]
    content: str


class AnthropicRequest(BaseModel):
    """Anthropic API request format."""

    model: str
    max_tokens: int = 1024
    messages: list[AnthropicMessage]
    system: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    metadata: dict | None = None

    model_config = {"extra": "allow"}


# Response Models


class AnthropicContentBlock(BaseModel):
    """Content block in Anthropic response."""

    type: Literal["text"] = "text"
    text: str = ""


class AnthropicUsage(BaseModel):
    """Token usage in Anthropic format."""

    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    """Anthropic API response format."""

    id: str = ""
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str = ""
    content: list[AnthropicContentBlock] = Field(default_factory=list)
    stop_reason: Literal["end_turn", "stop_sequence", "max_tokens", "tool_use"] | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)
