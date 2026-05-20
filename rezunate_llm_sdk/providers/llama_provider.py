"""
Llama (Meta) Provider — native API.

Transforms OpenAI format <-> Meta Llama native chat completion format.

Meta's native response uses a single ``completion_message`` with a typed
content block, plus a ``metrics`` array for token counts. This provider
maps those into the OpenAI ``choices``/``usage`` shape.
"""

import time
import uuid

import rezunate_llm_sdk.constants as constants
from rezunate_llm_sdk.models import (
    FINISH_REASON_MAP,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    FinishReason,
    Provider,
    ResponseMessage,
    Usage,
)
from rezunate_llm_sdk.providers.base import BaseProvider
from rezunate_llm_sdk.providers.endpoints import LLAMA_BASE_URL, LLAMA_CHAT_ENDPOINT
from rezunate_llm_sdk.providers.llama_models import (
    LlamaMessage,
    LlamaRequest,
    LlamaResponse,
)


class LlamaProvider(BaseProvider):
    """Llama provider against Meta's native chat completion API."""

    response_model = LlamaResponse

    @property
    def base_url(self) -> str:
        return LLAMA_BASE_URL

    @property
    def provider_name(self) -> Provider:
        return Provider.LLAMA

    def get_headers(self) -> dict[str, str]:
        return {
            constants.AUTHORIZATION_HEADER: f"Bearer {self.api_key}",
            constants.CONTENT_TYPE_HEADER: constants.APPLICATION_JSON,
        }

    def get_endpoint(self, model: str | None = None) -> str:
        return LLAMA_CHAT_ENDPOINT

    def transform_request(self, request: ChatCompletionRequest) -> LlamaRequest:
        """Transform OpenAI format request to Meta Llama native format.

        Key differences:
          - OpenAI uses ``max_tokens``; Meta uses ``max_completion_tokens``.
          - Meta supports ``repetition_penalty`` as a first-class field.
        """
        llama_messages = [
            LlamaMessage(role=msg.role.value, content=msg.content) for msg in request.messages
        ]

        return LlamaRequest(
            model=request.model,
            messages=llama_messages,
            max_completion_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=getattr(request, "top_k", None),
            repetition_penalty=getattr(request, "repetition_penalty", None),
            stream=request.stream,
            tools=getattr(request, "tools", None),
            response_format=getattr(request, "response_format", None),
        )

    def transform_response(
        self, response: LlamaResponse, model: str | None = None
    ) -> ChatCompletionResponse:
        """Transform Meta Llama native response to OpenAI format.

        Maps:
          - ``completion_message.content.text``    -> ``choices[0].message.content``
          - ``completion_message.stop_reason``     -> ``choices[0].finish_reason``
          - ``metrics[].value`` by metric name     -> ``usage.{prompt,completion,total}_tokens``
        """
        completion = response.completion_message
        content_text = completion.content.text if completion.content else ""
        finish_reason = FINISH_REASON_MAP.get(
            completion.stop_reason or "stop", FinishReason.STOP
        )

        choice = Choice(
            index=0,
            message=ResponseMessage(content=content_text),
            finish_reason=finish_reason,
        )

        # Unpack the metrics array into the OpenAI usage object.
        metric_values: dict[str, int] = {}
        for metric in response.metrics:
            metric_values[metric.metric] = int(metric.value)

        prompt_tokens = metric_values.get(constants.LLAMA_METRIC_PROMPT_TOKENS, 0)
        completion_tokens = metric_values.get(constants.LLAMA_METRIC_COMPLETION_TOKENS, 0)
        total_tokens = metric_values.get(
            constants.LLAMA_METRIC_TOTAL_TOKENS, prompt_tokens + completion_tokens
        )

        return ChatCompletionResponse(
            id=response.id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            provider=self.provider_name,
        )
