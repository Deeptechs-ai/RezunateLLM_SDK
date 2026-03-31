"""
Google Gemini-specific Pydantic models.

These models represent Google's Gemini API request/response format.
Used for type safety and validation in the Google provider.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Request Models


class GoogleContentBlock(BaseModel):
    """Content block containing text."""

    text: str = ""


class GoogleMessage(BaseModel):
    """Message in Google format with role and parts."""

    role: Literal["user", "model"]
    parts: list[GoogleContentBlock]


class GoogleSystemInstruction(BaseModel):
    """System instruction container."""

    parts: list[GoogleContentBlock]


class GoogleGenerationConfig(BaseModel):
    """Generation configuration parameters."""

    temperature: float | None = None
    maxOutputTokens: int | None = None
    topK: int | None = None

    model_config = {"extra": "allow"}


class GoogleRequest(BaseModel):
    """Google Gemini API request format."""

    contents: list[GoogleMessage]
    systemInstruction: GoogleSystemInstruction | None = None
    generationConfig: GoogleGenerationConfig | None = None
    safetySettings: Any | None = None
    tools: Any | None = None

    model_config = {"extra": "allow"}


# Response Models


class GoogleCandidate(BaseModel):
    """Candidate in Google response."""

    content: GoogleMessage | None = None
    finishReason: Literal["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"] | None = None


class GoogleUsage(BaseModel):
    """Token usage in Google format."""

    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    totalTokenCount: int = 0


class GoogleResponse(BaseModel):
    """Google Gemini API response format."""

    candidates: list[GoogleCandidate] = Field(default_factory=list)
    usageMetadata: GoogleUsage = Field(default_factory=GoogleUsage)
