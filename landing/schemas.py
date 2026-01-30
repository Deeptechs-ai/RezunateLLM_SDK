"""Pydantic schemas for request/response validation."""

from datetime import datetime

from pydantic import BaseModel, EmailStr


class WaitlistSignupRequest(BaseModel):
    """Schema for waitlist signup request."""

    email: EmailStr
    name: str | None = None
    company: str | None = None
    use_case: str | None = None


class WaitlistSignupResponse(BaseModel):
    """Schema for successful waitlist signup response."""

    id: int
    email: str
    name: str | None
    company: str | None
    use_case: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    detail: str


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str


class ChatMessage(BaseModel):
    """Schema for a chat message."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Schema for chat demo request."""

    messages: list[ChatMessage]
    session_id: str


class ChatResponse(BaseModel):
    """Schema for chat demo response."""

    content: str
    model: str
    usage: dict | None = None
