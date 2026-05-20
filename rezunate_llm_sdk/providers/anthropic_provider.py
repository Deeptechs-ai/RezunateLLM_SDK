"""
Anthropic Provider.
Transforms OpenAI format to Anthropic format
"""

import time
import uuid
from typing import Any

import rezunate_llm_sdk.constants as constants
from rezunate_llm_sdk.models import (
    FINISH_REASON_MAP,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    FinishReason,
    Provider,
    ResponseMessage,
    Role,
    Usage,
)
from rezunate_llm_sdk.providers.anthropic_models import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
)
from rezunate_llm_sdk.providers.base import BaseProvider, StreamState
from rezunate_llm_sdk.providers.endpoints import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_DEFAULT_VERSION,
    ANTHROPIC_MESSAGES_ENDPOINT,
    get_url,
)


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
    def provider_name(self) -> Provider:
        return Provider.ANTHROPIC

    def get_headers(self) -> dict[str, str]:
        return {
            constants.API_KEY_HEADER: self.api_key,
            constants.CONTENT_TYPE_HEADER: constants.APPLICATION_JSON,
            constants.ANTHROPIC_VERSION_HEADER: ANTHROPIC_DEFAULT_VERSION,
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
            if role == Role.SYSTEM:
                system_content = content
            else:
                # Map OpenAI roles to Anthropic roles
                anthropic_role = "assistant" if role == Role.ASSISTANT else "user"
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
        finish_reason = FINISH_REASON_MAP.get(anthropic_response.stop_reason, FinishReason.STOP)

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

    # ---- streaming hooks (driven by BaseProvider.stream) ----------------

    def _build_stream_request(
        self, request: ChatCompletionRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        body = self.transform_request(request).model_dump(exclude_none=True)
        body["stream"] = True
        return get_url(self.base_url, self.get_endpoint()), body, self.get_headers()

    def _is_stream_terminator(self, event: str, data: str) -> bool:
        return event == "message_stop"

    def _translate_frame(
        self, event: str, data: str, state: StreamState
    ) -> ChatCompletionChunk | None:
        if event in ("ping", "content_block_start", "content_block_stop"):
            return None

        payload = self._parse_json_frame(data) or {}

        if event == "message_start":
            message = payload.get("message", {})
            state.id = message.get("id") or state.id
            state.model = message.get("model") or state.model
            state.input_tokens = message.get("usage", {}).get("input_tokens", 0)
            return self._make_chunk(state, delta=ChoiceDelta(role=Role.ASSISTANT, content=""))

        if event == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta" and (text := delta.get("text", "")):
                return self._make_chunk(state, delta=ChoiceDelta(content=text))
            return None

        if event == "message_delta":
            stop_reason = payload.get("delta", {}).get("stop_reason")
            finish = FINISH_REASON_MAP.get(stop_reason, FinishReason.STOP) if stop_reason else None
            output_tokens = payload.get("usage", {}).get("output_tokens", 0)
            usage = Usage(
                prompt_tokens=state.input_tokens,
                completion_tokens=output_tokens,
                total_tokens=state.input_tokens + output_tokens,
            )
            return self._make_chunk(state, finish_reason=finish, usage=usage)

        if event == "error":
            message = payload.get("error", {}).get("message") or "anthropic stream error"
            return self._error_chunk(RuntimeError(message), state.model)

        return None
