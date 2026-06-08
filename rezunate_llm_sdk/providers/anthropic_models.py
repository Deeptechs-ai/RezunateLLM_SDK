"""
Anthropic-specific Pydantic models.

These models represent Anthropic's API request/response format.
Used for type safety and validation in the Anthropic provider.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# Content blocks (used in both request and response).


class AnthropicTextBlock(BaseModel):
    """A text content block."""

    type: Literal["text"] = "text"
    text: str = ""


class AnthropicToolUseBlock(BaseModel):
    """A ``tool_use`` block: the assistant requesting a tool call.

    Appears in Anthropic responses and in the request when echoing the prior
    assistant turn back to the model on the next round-trip.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    # Tool argument payload defined by the user's tool.
    input: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    """A ``tool_result`` block: the tool's output, sent under a user-role message."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[AnthropicTextBlock] = ""


AnthropicContentBlock = Annotated[
    AnthropicTextBlock | AnthropicToolUseBlock | AnthropicToolResultBlock,
    Field(discriminator="type"),
]


# Tool surface


class AnthropicTool(BaseModel):
    """Anthropic-shaped tool definition."""

    name: str
    description: str = ""
    # Caller-supplied JSON Schema body.
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicAutoToolChoice(BaseModel):
    """Model decides whether to call a tool."""

    type: Literal["auto"] = "auto"


class AnthropicAnyToolChoice(BaseModel):
    """Model must call exactly one tool."""

    type: Literal["any"] = "any"


class AnthropicSpecificToolChoice(BaseModel):
    """Model must call this specific tool."""

    type: Literal["tool"] = "tool"
    name: str


AnthropicToolChoice = Annotated[
    AnthropicAutoToolChoice | AnthropicAnyToolChoice | AnthropicSpecificToolChoice,
    Field(discriminator="type"),
]


# Request Models


class AnthropicMessage(BaseModel):
    """Message in Anthropic format.

    ``content`` is either a plain string or a list of typed
    content blocks (text + tool_use + tool_result mixes).
    """

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicRequest(BaseModel):
    """Anthropic API request format."""

    model: str
    max_tokens: int = 1024
    messages: list[AnthropicMessage]
    system: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    metadata: dict | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None

    model_config = {"extra": "allow"}


# Response Models


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
