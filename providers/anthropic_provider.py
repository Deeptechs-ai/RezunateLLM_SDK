"""
Anthropic Provider.
Transforms OpenAI format to Anthropic format
"""

import time
import uuid
from typing import Any

from providers.base import BaseProvider


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
        anthropic_request = {
            "model": request.get("model"),
            "max_tokens": request.get("max_tokens", 1024),  # Anthropic requires this
        }

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
                anthropic_messages.append({"role": anthropic_role, "content": content})

        anthropic_request["messages"] = anthropic_messages

        # Add system message if present
        if system_content:
            anthropic_request["system"] = system_content

        # Map optional parameters
        if "temperature" in request:
            anthropic_request["temperature"] = request["temperature"]

        # Pass through Anthropic-specific parameters
        # These are params that Anthropic supports but OpenAI doesn't
        anthropic_specific_params = ["top_k", "metadata"]
        for param in anthropic_specific_params:
            if param in request:
                anthropic_request[param] = request[param]

        return anthropic_request

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
        # Extract content from Anthropic response
        content = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        # Map Anthropic stop_reason to OpenAI finish_reason
        stop_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        finish_reason = stop_reason_map.get(response.get("stop_reason", "end_turn"), "stop")

        # Build OpenAI format response
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        openai_response = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

        return openai_response
