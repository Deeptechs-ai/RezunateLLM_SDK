"""
Google Gemini-specific Pydantic models.

These models represent Google's Gemini API request/response format.
Used for type safety and validation in the Google provider.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GoogleFunctionCall(BaseModel):
    """A ``functionCall`` part: the model requesting a tool call."""

    name: str
    # Arbitrary tool argument payload defined by the user's tool.
    args: dict[str, Any] = Field(default_factory=dict)


class GoogleFunctionResponse(BaseModel):
    """A ``functionResponse`` part: the result of a tool execution."""

    name: str
    # Arbitrary tool result payload defined by the user's tool.
    response: dict[str, Any] = Field(default_factory=dict)


# Request Models


class GoogleContentBlock(BaseModel):
    """A single ``part`` inside a Gemini ``Content``.

    Gemini parts are tagged unions in the wire protocol — ``text``,
    ``functionCall``, ``functionResponse``, etc. Only one of these fields is set
    per part.
    """

    model_config = ConfigDict(extra="allow")

    text: str | None = None
    functionCall: GoogleFunctionCall | None = None
    functionResponse: GoogleFunctionResponse | None = None


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


# Tool surface


class GoogleFunctionDeclaration(BaseModel):
    """Gemini-shaped tool function declaration."""

    name: str
    description: str = ""
    # Caller-supplied JSON Schema body.
    parameters: dict[str, Any] = Field(default_factory=dict)


class GoogleTool(BaseModel):
    """Outer ``tools`` entry containing a batch of function declarations."""

    functionDeclarations: list[GoogleFunctionDeclaration] = Field(default_factory=list)


class GoogleFunctionCallingConfig(BaseModel):
    """``toolConfig.functionCallingConfig`` — mode + optional allowlist."""

    mode: Literal["AUTO", "NONE", "ANY"] = "AUTO"
    allowedFunctionNames: list[str] | None = None


class GoogleToolConfig(BaseModel):
    """Outer ``toolConfig`` payload."""

    functionCallingConfig: GoogleFunctionCallingConfig


class GoogleRequest(BaseModel):
    """Google Gemini API request format."""

    contents: list[GoogleMessage]
    systemInstruction: GoogleSystemInstruction | None = None
    generationConfig: GoogleGenerationConfig | None = None
    safetySettings: Any | None = None
    tools: list[GoogleTool] | None = None
    toolConfig: GoogleToolConfig | None = None

    model_config = {"extra": "allow"}


# Response Models


class GoogleCandidate(BaseModel):
    """Candidate in Google response."""

    content: GoogleMessage | None = None
    finishReason: (
        Literal["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", "FUNCTION_CALL"] | None
    ) = None


class GoogleUsage(BaseModel):
    """Token usage in Google format."""

    promptTokenCount: int = 0
    candidatesTokenCount: int = 0
    totalTokenCount: int = 0


class GoogleResponse(BaseModel):
    """Google Gemini API response format."""

    candidates: list[GoogleCandidate] = Field(default_factory=list)
    usageMetadata: GoogleUsage = Field(default_factory=GoogleUsage)
