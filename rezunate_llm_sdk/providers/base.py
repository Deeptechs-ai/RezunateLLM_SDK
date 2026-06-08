"""
Base Provider Class.
All providers inherit from this class.
"""

import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import requests
from pydantic import BaseModel

import rezunate_llm_sdk.constants as constants
from rezunate_llm_sdk.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceChunk,
    ChoiceDelta,
    ErrorInfo,
    FinishReason,
    Provider,
    Usage,
)
from rezunate_llm_sdk.providers.endpoints import get_url
from rezunate_llm_sdk.streaming.sse_parser import parse_sse_lines


@dataclass
class StreamState:
    """Per-stream mutable state passed to translator hooks.

    Common fields (``id``, ``model``, ``created``) cover the majority of
    providers. Provider-specific fields (``input_tokens``, ``role_sent``)
    are present here so providers that need them can use them directly
    without subclassing.
    """

    id: str
    model: str | None = None
    created: int = 0
    input_tokens: int = 0
    role_sent: bool = False


class BaseProvider(ABC):
    """
    Abstract base class for all providers.
    Each provider must implement these methods.
    """

    response_model: type[BaseModel] | None = None

    def __init__(
        self,
        api_key: str,
        max_retries: int = constants.DEFAULT_MAX_RETRIES,
        retry_delay: float = constants.DEFAULT_RETRY_DELAY,
        timeout: float = 60.0,
        **kwargs,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for the provider's API."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> Provider:
        """Return the provider name."""
        pass

    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Return the headers required for API calls."""
        pass

    @abstractmethod
    def get_endpoint(self, model: str | None = None) -> str:
        """Return the chat completion endpoint."""
        pass

    @abstractmethod
    def transform_request(self, request: ChatCompletionRequest) -> BaseModel:
        """
        Transform OpenAI format request to provider's format.
        For OpenAI provider, this returns the request unchanged.
        """
        pass

    @abstractmethod
    def transform_response(
        self, response: BaseModel, model: str | None = None
    ) -> ChatCompletionResponse:
        """
        Transform provider's response Pydantic model to OpenAI format.
        """
        pass

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        # Exponential backoff: delay * 2^attempt + random jitter
        backoff = self.retry_delay * (2**attempt)
        jitter = random.uniform(0, backoff * 0.1)
        return backoff + jitter

    def _is_retryable(self, status_code: int | None) -> bool:
        """Check if the error is retryable based on status code."""
        return status_code in constants.RETRYABLE_STATUS_CODES

    def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Execute chat completion with retry logic.

        1. Transform request to provider format
        2. Execute request (with retries)
        3. Transform response to OpenAI format
        4. Return response
        """
        # Transform request to provider format
        provider_request = self.transform_request(request)

        # Extract model for endpoint and response transformation
        model = request.model

        try:
            # Execute the request (subclasses can override this to use an SDK)
            provider_response = self._execute_request(provider_request, model)

            # Success - transform and return
            openai_response = self.transform_response(provider_response, model)
            openai_response.provider = self.provider_name
            return openai_response

        except Exception as e:
            # Return error in consistent OpenAI format
            return self._handle_error(e, model)

    def _execute_request(self, provider_request: BaseModel, model: str | None = None) -> Any:
        """
        Execute the request with retry logic.
        Default implementation uses the requests library.
        Subclasses can override this to use their own SDKs.
        """
        # Build full URL
        url = get_url(self.base_url, self.get_endpoint(model))
        last_error = None
        attempt = 0

        # Retry loop
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=self.get_headers(),
                    json=provider_request.model_dump(exclude_none=True),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                provider_response_dict = response.json()

                # Validate response using provider's model
                if self.response_model:
                    return self.response_model.model_validate(provider_response_dict)
                return provider_response_dict

            except requests.exceptions.RequestException as e:
                last_error = e
                status_code = (
                    getattr(e.response, "status_code", None) if hasattr(e, "response") else None
                )

                # Check if we should retry
                is_timeout = isinstance(e, requests.exceptions.Timeout)
                is_connection_error = isinstance(e, requests.exceptions.ConnectionError)

                if attempt < self.max_retries and (
                    self._is_retryable(status_code) or is_timeout or is_connection_error
                ):
                    backoff = self._calculate_backoff(attempt)
                    time.sleep(backoff)
                    continue

                # No more retries
                raise last_error from e

        # Should not reach here if max_retries >= 0
        raise last_error if last_error else RuntimeError("Request failed without error")

    def _handle_error(self, error: Exception, model: str | None = None) -> ChatCompletionResponse:
        """Centralized error handling for all providers."""
        status_code = None
        if hasattr(error, "response") and error.response is not None:
            status_code = getattr(error.response, "status_code", None)

        return ChatCompletionResponse(
            id=None,
            created=int(time.time()),
            model=model,
            choices=[],
            usage=Usage(),
            provider=self.provider_name,
            error=ErrorInfo(
                message=str(error),
                type="api_error",
                code=status_code,
            ),
        )

    # Streaming
    def stream(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
        """Streaming chat completion.

        Default implementation drives the request through SSE. Override
        for SDK-based streaming.
        """
        state = self._new_stream_state(request)
        try:
            url, body, headers = self._build_stream_request(request)
            lines = self._stream_http_lines(url, body, headers)
            for event, data in parse_sse_lines(lines):
                if self._is_stream_terminator(event, data):
                    return
                if (chunk := self._translate_frame(event, data, state)) is not None:
                    yield chunk
        except Exception as e:
            yield self._error_chunk(e, request.model)

    # ---- hooks (override in native-protocol providers) ------------------

    def _new_stream_state(self, request: ChatCompletionRequest) -> StreamState:
        """Per-stream mutable state. Defaults to a fresh ``StreamState``."""
        return StreamState(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            model=request.model,
            created=int(time.time()),
        )

    def _build_stream_request(
        self, request: ChatCompletionRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Return ``(url, body, headers)`` for the streaming POST."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _build_stream_request")

    def _translate_frame(
        self, event: str, data: str, state: StreamState
    ) -> ChatCompletionChunk | None:
        """Translate one SSE frame into a chunk, or ``None`` to skip it."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _translate_frame")

    def _is_stream_terminator(self, event: str, data: str) -> bool:
        """Return ``True`` when this frame ends the stream."""
        return False

    # ---- shared helpers --------------------------------------------------

    @staticmethod
    def _parse_json_frame(data: str) -> dict | None:
        """Parse a JSON SSE frame payload. Returns ``None`` if empty or malformed."""
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    def _make_chunk(
        self,
        state: StreamState,
        delta: ChoiceDelta | None = None,
        finish_reason: FinishReason | None = None,
        usage: Usage | None = None,
    ) -> ChatCompletionChunk:
        """Build a ``ChatCompletionChunk`` from stream state and a delta."""
        return ChatCompletionChunk(
            id=state.id,
            created=state.created,
            model=state.model,
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=delta or ChoiceDelta(),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
            provider=self.provider_name,
        )

    def _error_chunk(self, error: Exception, model: str | None = None) -> ChatCompletionChunk:
        """Terminal-error chunk mirroring the shape of ``_handle_error``."""
        status_code = None
        if hasattr(error, "response") and error.response is not None:
            status_code = getattr(error.response, "status_code", None)

        return ChatCompletionChunk(
            id=None,
            created=int(time.time()),
            model=model,
            choices=[],
            provider=self.provider_name,
            error=ErrorInfo(
                message=str(error),
                type="api_error",
                code=status_code,
            ),
        )

    def _stream_http_lines(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Open an httpx stream and yield decoded text lines."""
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST", url, json=body, headers=headers or self.get_headers()
            ) as response:
                response.raise_for_status()
                yield from response.iter_lines()
