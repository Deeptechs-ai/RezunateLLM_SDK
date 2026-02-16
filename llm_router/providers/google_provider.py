"""
Google (Gemini) Provider.
Transforms OpenAI format <-> Google Gemini format.
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
from llm_router.providers.base import BaseProvider
from llm_router.providers.google_models import (
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

    response_model = GoogleResponse

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

    def transform_request(self, request: ChatCompletionRequest) -> GoogleRequest:
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
        system_content = None
        google_messages = []

        for msg in request.messages:
            role = msg.role
            content = msg.content

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
        if (
            request.temperature is not None
            or request.max_tokens is not None
            or getattr(request, "top_k", None) is not None
        ):
            generation_config = GoogleGenerationConfig(
                temperature=request.temperature,
                maxOutputTokens=request.max_tokens,
                topK=getattr(request, "top_k", None),
            )

        # Build Google request using pydantic model
        google_request = GoogleRequest(
            contents=google_messages,
            systemInstruction=system_instruction,
            generationConfig=generation_config,
            safetySettings=getattr(request, "safety_settings", None),
            tools=getattr(request, "tools", None),
        )

        return google_request

    def transform_response(
        self, response: GoogleResponse, model: str | None = None
    ) -> ChatCompletionResponse:
        """Transform Google Gemini format response to OpenAI format.

        Converts candidates to choices, maps 'model' role to 'assistant',
        translates finish reasons, and normalizes usage metadata.

        Args:
            response: Google Gemini response Pydantic model.
            model: Model name to include in the response.

        Returns:
            OpenAI-formatted response with id, object, created, model, choices, and usage.
        """
        # google_response is already validated by BaseProvider
        google_response = response

        # Map Google finish reason to OpenAI
        finish_reason_map = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop",
        }

        # Extract content from candidates using Pydantic models
        choices = []
        for idx, candidate in enumerate(google_response.candidates):
            content = ""
            if candidate.content:
                for part in candidate.content.parts:
                    content += part.text

            finish_reason = finish_reason_map.get(candidate.finishReason or "STOP", "stop")

            choices.append(
                Choice(
                    index=idx,
                    message=ResponseMessage(content=content),
                    finish_reason=finish_reason,
                )
            )

        # Extract usage metadata
        usage_meta = google_response.usageMetadata
        prompt_tokens = usage_meta.promptTokenCount
        completion_tokens = usage_meta.candidatesTokenCount
        total_tokens = usage_meta.totalTokenCount or (prompt_tokens + completion_tokens)

        openai_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            provider=self.provider_name,
        )

        return openai_response
