"""
Base Provider Class.
All providers inherit from this class.
"""

import random
import time
from abc import ABC, abstractmethod

import requests
from pydantic import BaseModel

from llm_router.models import ChatCompletionRequest, ChatCompletionResponse, ErrorInfo, Usage

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class BaseProvider(ABC):
    """
    Abstract base class for all providers.
    Each provider must implement these methods.
    """

    response_model: type[BaseModel] | None = None

    def __init__(
        self,
        api_key: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        timeout: float = 60.0,
        **kwargs,
    ):
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
    def provider_name(self) -> str:
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
        return status_code in RETRYABLE_STATUS_CODES

    def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Execute chat completion with retry logic.

        1. Transform request to provider format
        2. Call provider API (with retries)
        3. Transform response to OpenAI format
        4. Return response
        """
        # Transform request to provider format
        provider_request = self.transform_request(request)

        # Extract model for endpoint and response transformation
        model = request.model

        # Build full URL
        url = f"{self.base_url}{self.get_endpoint(model)}"

        last_error = None

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
                    provider_response = self.response_model.model_validate(provider_response_dict)
                else:
                    # Fallback for providers without models (though we aim to have models for all)
                    provider_response = provider_response_dict

                # Success - transform and return
                openai_response = self.transform_response(provider_response, model)
                openai_response.provider = self.provider_name
                return openai_response

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

                # No more retries - return error
                break

        # Return error in consistent OpenAI format using Pydantic models
        error_code = (
            getattr(last_error.response, "status_code", None)
            if hasattr(last_error, "response")
            else None
        )

        error_response = ChatCompletionResponse(
            id=None,
            created=int(time.time()),
            model=model,
            choices=[],
            usage=Usage(),
            provider=self.provider_name,
            error=ErrorInfo(
                message=str(last_error),
                type="api_error",
                code=error_code,
                retries_attempted=attempt,
            ),
        )

        return error_response
