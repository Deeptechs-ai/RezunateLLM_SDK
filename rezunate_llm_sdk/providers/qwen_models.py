"""
Qwen (Alibaba DashScope) native API Pydantic models.

These represent the DashScope text-generation API request/response format
(NOT the OpenAI-compatible mode). Used for type safety and validation in
the native Qwen provider.

Reference: Alibaba Model Studio DashScope text-generation API.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Request Models


class QwenMessage(BaseModel):
    """Message in DashScope native format."""

    role: Literal["system", "user", "assistant"]
    content: str


class QwenInput(BaseModel):
    """Input wrapper for DashScope native request."""

    messages: list[QwenMessage]


class QwenParameters(BaseModel):
    """Generation parameters for DashScope native request."""

    result_format: Literal["message", "text"] = "message"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    enable_search: bool | None = None
    repetition_penalty: float | None = None

    model_config = {"extra": "allow"}


class QwenRequest(BaseModel):
    """DashScope native generation request."""

    model: str
    input: QwenInput
    parameters: QwenParameters | None = None

    model_config = {"extra": "allow"}


# Response Models


class QwenResponseMessage(BaseModel):
    """Message inside a DashScope response choice."""

    role: Literal["assistant"] = "assistant"
    content: str = ""


class QwenChoice(BaseModel):
    """A single choice in the DashScope response output."""

    finish_reason: str | None = None
    message: QwenResponseMessage = Field(default_factory=QwenResponseMessage)


class QwenOutput(BaseModel):
    """Output container in the DashScope response."""

    choices: list[QwenChoice] = Field(default_factory=list)
    # Legacy `text` and `finish_reason` fields present when result_format="text"
    text: str | None = None
    finish_reason: str | None = None


class QwenUsage(BaseModel):
    """Token usage from DashScope response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class QwenResponse(BaseModel):
    """DashScope native generation response."""

    output: QwenOutput = Field(default_factory=QwenOutput)
    usage: QwenUsage = Field(default_factory=QwenUsage)
    request_id: str | None = None

    model_config = {"extra": "allow"}
