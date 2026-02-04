"""FastAPI application entry point for the landing page."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from landing.config import settings
from landing.database import init_db
from landing.routes import api


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler for startup and shutdown events."""
    await init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Unified Python SDK for chat completions across multiple AI providers",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api.router)
