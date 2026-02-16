"""
Anthropic Provider.
Transforms OpenAI format to Anthropic format
"""

import time
import uuid

from llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
    Usage,
)
from llm_router.providers.anthropic_models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
)
from llm_router.providers.base import BaseProvider
from llm_router.providers.endpoints import ANTHROPIC_BASE_URL, ANTHROPIC_MESSAGES_ENDPOINT


class AnthropicProvider(BaseProvider):
    """
    Anthropic Provider implementation.
    Handles transformation between OpenAI and Anthropic formats.
    """

    response_model = AnthropicResponse

    @property
    def base_url(self) -> str:
        return ANTHROPIC_BASE_URL

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def get_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def get_endpoint(self, model: str | None = None) -> str:
        return ANTHROPIC_MESSAGES_ENDPOINT

    def transform_request(self, request: ChatCompletionRequest) -> AnthropicRequest:
        """
        Transform OpenAI format request to Anthropic format.

        Key differences:
        - OpenAI: system message in messages array
        - Anthropic: system message is separate parameter
        - OpenAI: max_tokens optional
        - Anthropic: max_tokens required
        """
        # Extract system message and regular messages
        system_content = None
        anthropic_messages = []

        for msg in request.messages:
            role = msg.role
            content = msg.content

            # Anthropic handles system separately
            if role == "system":
                system_content = content
            else:
                # Map OpenAI roles to Anthropic roles
                anthropic_role = "assistant" if role == "assistant" else "user"
                anthropic_messages.append(AnthropicMessage(role=anthropic_role, content=content))

        # Build Anthropic request using pydantic model
        anthropic_request = AnthropicRequest(
            model=request.model,
            max_tokens=request.max_tokens or 1024,
            messages=anthropic_messages,
            system=system_content,
            temperature=request.temperature,
            top_k=getattr(request, "top_k", None),  # top_k is extra field in ChatCompletionRequest
            metadata=getattr(request, "metadata", None),
        )

        return anthropic_request

    def transform_response(
        self, response: AnthropicResponse, model: str | None = None
    ) -> ChatCompletionResponse:
        """
        Transform Anthropic format response to OpenAI format.
        """
        # anthropic_response is already validated by BaseProvider
        anthropic_response = response

        # Extract content from Anthropic response
        content = ""
        for block in anthropic_response.content:
            if block.type == "text":
                content += block.text

        # Map Anthropic stop_reason to OpenAI finish_reason
        stop_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        finish_reason = stop_reason_map.get(anthropic_response.stop_reason or "end_turn", "stop")

        # Build OpenAI format response using Pydantic models
        input_tokens = anthropic_response.usage.input_tokens
        output_tokens = anthropic_response.usage.output_tokens

        openai_response = ChatCompletionResponse(
            id=anthropic_response.id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=anthropic_response.model or model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            provider=self.provider_name,
        )

        return openai_response
