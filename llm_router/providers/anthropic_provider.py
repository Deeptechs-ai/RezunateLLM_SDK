"""
Anthropic Provider.
Transforms OpenAI format to Anthropic format
"""

import time
import uuid
from typing import Any

from pydantic import ValidationError

from llm_router.models import ChatCompletionResponse, Choice, ResponseMessage, Usage
from llm_router.providers.anthropic_models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
)
from llm_router.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """
    Anthropic Provider implementation.
    Handles transformation between OpenAI and Anthropic formats.
    """

    @property
    def base_url(self) -> str:
        return "https://api.anthropic.com/v1"

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
        return "/messages"

    def transform_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Transform OpenAI format request to Anthropic format.

        Key differences:
        - OpenAI: system message in messages array
        - Anthropic: system message is separate parameter
        - OpenAI: max_tokens optional
        - Anthropic: max_tokens required
        """
        # Extract system message and regular messages
        messages = request.get("messages", [])
        system_content = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Anthropic handles system separately
            if role == "system":
                system_content = content
            else:
                # Map OpenAI roles to Anthropic roles
                anthropic_role = "assistant" if role == "assistant" else "user"
                anthropic_messages.append(AnthropicMessage(role=anthropic_role, content=content))

        # Build Anthropic request using pydantic model
        anthropic_request = AnthropicRequest(
            model=request.get("model"),
            max_tokens=request.get("max_tokens", 1024),
            messages=anthropic_messages,
            system=system_content,
            temperature=request.get("temperature"),
            top_k=request.get("top_k"),
            metadata=request.get("metadata"),
        )

        return anthropic_request.model_dump(exclude_none=True)

    def transform_response(
        self, response: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        """
        Transform Anthropic format response to OpenAI format.

        Anthropic response:
        {
            "id": "msg_...",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "..."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": X, "output_tokens": Y}
        }

        OpenAI format:
        {
            "id": "chatcmpl-...",
            "object": "chat.completion",
            "created": timestamp,
            "model": "...",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "..."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}
        }
        """
        # Parse Anthropic response using pydantic model with error handling
        try:
            anthropic_response = AnthropicResponse.model_validate(response)
        except ValidationError as e:
            raise ValueError(f"Invalid Anthropic response: {e}") from e

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
            model=anthropic_response.model,
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

        return openai_response.model_dump(exclude_none=True)
