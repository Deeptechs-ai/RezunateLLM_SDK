"""
OpenAI Provider.
No transformation needed - OpenAI format is the standard.
"""

from typing import Any

from models import ChatCompletionRequest
from providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    OpenAI Provider implementation.
    Since OpenAI format is our standard, no transformation is needed.
    """

    @property
    def base_url(self) -> str:
        return "https://api.openai.com/v1"

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def get_endpoint(self, model: str = None) -> str:
        return "/chat/completions"

    def transform_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and pass through - already in OpenAI format."""
        openai_request = ChatCompletionRequest.model_validate(request)
        return openai_request.model_dump(exclude_none=True)

    def transform_response(self, response: dict[str, Any], model: str = None) -> dict[str, Any]:
        """No transformation needed - already in OpenAI format."""
        return response
