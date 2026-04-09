"""
OpenAI Provider.
Uses official OpenAI SDK.
"""

from typing import Any

from openai import OpenAI
import rezunate_llm_sdk.constants as constants

from rezunate_llm_sdk.models import ChatCompletionRequest, ChatCompletionResponse, Provider
from rezunate_llm_sdk.providers.base import BaseProvider
from rezunate_llm_sdk.providers.endpoints import OPENAI_BASE_URL, OPENAI_CHAT_ENDPOINT


class OpenAIProvider(BaseProvider):
    """
    OpenAI Provider implementation using the official SDK.
    """

    response_model = ChatCompletionResponse

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            max_retries=self.max_retries,
        )

    @property
    def base_url(self) -> str:
        return OPENAI_BASE_URL

    @property
    def provider_name(self) -> Provider:
        return Provider.OPENAI

    def get_headers(self) -> dict[str, str]:
        """Not used when using SDK, but kept for interface consistency."""

        return {
            constants.AUTHORIZATION_HEADER: f"Bearer {self.api_key}",
            constants.CONTENT_TYPE_HEADER: constants.APPLICATION_JSON,
        }

    def get_endpoint(self, model: str | None = None) -> str:
        """Not used when using SDK."""
        return OPENAI_CHAT_ENDPOINT

    def transform_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Validate and pass through - already in OpenAI format."""
        return request

    def _execute_request(
        self, provider_request: ChatCompletionRequest, model: str | None = None
    ) -> Any:
        """Execute the request using the OpenAI SDK."""
        params = provider_request.model_dump(exclude_none=True)
        return self.client.chat.completions.create(**params, timeout=self.timeout)

    def transform_response(self, response: Any, model: str | None = None) -> ChatCompletionResponse:
        """Convert SDK response to our internal model."""
        if hasattr(response, "model_dump"):
            return ChatCompletionResponse.model_validate(response.model_dump())
        return response
