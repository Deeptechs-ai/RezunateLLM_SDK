"""
Google (Gemini) Provider.
Transforms OpenAI format <-> Google Gemini format.
"""

import time
import uuid
from typing import Any

from providers.base import BaseProvider
from providers.google_models import (
    GoogleContentBlock,
    GoogleGenerationConfig,
    GoogleMessage,
    GoogleRequest,
    GoogleResponse,
    GoogleSystemInstruction,
)


class GoogleProvider(BaseProvider):
    """
    Google Gemini Provider implementation.
    Handles transformation between OpenAI and Google Gemini formats.
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)

    @property
    def base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    @property
    def provider_name(self) -> str:
        return "google"

    def get_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    def get_endpoint(self, model: str = None) -> str:
        return f"/models/{model}:generateContent"

    def transform_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Transform OpenAI format request to Google Gemini format.

        Converts messages to Google's contents format, maps 'assistant' role to 'model',
        extracts system messages to systemInstruction, and maps generation parameters.

        Args:
            request: OpenAI-formatted request containing messages, model, and optional
                parameters like temperature, max_tokens, top_k, safety_settings, and tools.

        Returns:
            Google Gemini-formatted request with contents, systemInstruction (if present),
            and generationConfig.
        """
        # Extract system message and regular messages
        messages = request.get("messages", [])
        system_content = None
        google_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            else:
                google_role = "model" if role == "assistant" else "user"
                google_messages.append(
                    GoogleMessage(role=google_role, parts=[GoogleContentBlock(text=content)])
                )

        # Build system instruction if present
        system_instruction = None
        if system_content:
            system_instruction = GoogleSystemInstruction(
                parts=[GoogleContentBlock(text=system_content)]
            )

        # Build generation config if any params present
        generation_config = None
        if any(k in request for k in ["temperature", "max_tokens", "top_k"]):
            generation_config = GoogleGenerationConfig(
                temperature=request.get("temperature"),
                maxOutputTokens=request.get("max_tokens"),
                topK=request.get("top_k"),
            )

        # Build Google request using pydantic model
        google_request = GoogleRequest(
            contents=google_messages,
            systemInstruction=system_instruction,
            generationConfig=generation_config,
            safetySettings=request.get("safety_settings"),
            tools=request.get("tools"),
        )

        return google_request.model_dump(exclude_none=True)

    def transform_response(self, response: dict[str, Any], model: str = None) -> dict[str, Any]:
        """Transform Google Gemini format response to OpenAI format.

        Converts candidates to choices, maps 'model' role to 'assistant',
        translates finish reasons, and normalizes usage metadata.

        Args:
            response: Google Gemini response containing candidates and usageMetadata.
            model: Model name to include in the response.

        Returns:
            OpenAI-formatted response with id, object, created, model, choices, and usage.
        """
        # Parse Google response using pydantic model
        google_response = GoogleResponse.model_validate(response)

        # Map Google finish reason to OpenAI
        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop",
        }

        # Extract content from candidates
        choices = []
        for idx, candidate in enumerate(google_response.candidates):
            content = ""
            if candidate.content:
                for part in candidate.content.parts:
                    content += part.text

            finish_reason = finish_reason_map.get(candidate.finishReason or "STOP", "stop")

            choices.append(
                {
                    "index": idx,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            )

        # Extract usage metadata
        usage = google_response.usageMetadata
        prompt_tokens = usage.promptTokenCount
        completion_tokens = usage.candidatesTokenCount
        total_tokens = usage.totalTokenCount or (prompt_tokens + completion_tokens)

        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        return openai_response
