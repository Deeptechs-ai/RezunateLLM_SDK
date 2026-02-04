"""API routes for waitlist signup and chat demo."""

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway import chat_complete
from landing.config import settings
from landing.database import get_session
from landing.models import WaitlistSignup
from landing.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    WaitlistSignupRequest,
    WaitlistSignupResponse,
)
from models import ChatCompletionRequest, Message

router = APIRouter(prefix="/api", tags=["api"])

# Simple in-memory rate limiting by session ID
_session_request_counts: dict[str, int] = defaultdict(int)


@router.post(
    "/waitlist",
    response_model=WaitlistSignupResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
)
async def signup_waitlist(
    data: WaitlistSignupRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> WaitlistSignup:
    """Sign up for the waitlist."""
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    signup = WaitlistSignup(
        email=data.email,
        name=data.name,
        company=data.company,
        use_case=data.use_case,
        ip_address=ip_address,
        user_agent=user_agent[:500] if user_agent else None,
    )

    session.add(signup)

    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        existing = await session.execute(
            select(WaitlistSignup).where(WaitlistSignup.email == data.email)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This email is already registered on the waitlist.",
            ) from None
        raise

    await session.refresh(signup)
    return signup


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def chat_demo(data: ChatRequest) -> ChatResponse:
    """Interactive chat demo endpoint."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat demo is not configured.",
        )

    # Rate limiting
    if _session_request_counts[data.session_id] >= settings.demo_rate_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Demo limit reached ({settings.demo_rate_limit} messages). Join the waitlist for full access!",
        )

    _session_request_counts[data.session_id] += 1

    messages = [Message(role=m.role, content=m.content) for m in data.messages]

    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
    )

    try:
        response = chat_complete(
            provider="openai",
            api_key=settings.openai_api_key,
            request=request,
        )
    except Exception as e:
        _session_request_counts[data.session_id] -= 1  # Don't count failed requests
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chat service error: {str(e)}",
        ) from e

    return ChatResponse(
        content=response.choices[0].message.content,
        model=response.model,
        usage=response.usage.model_dump() if response.usage else None,
    )
