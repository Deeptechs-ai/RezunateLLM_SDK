"""
Llama (Meta) Provider — native API.

Transforms OpenAI format <-> Meta Llama native chat completion format.

Meta's native response uses a single ``completion_message`` with a typed
content block, plus a ``metrics`` array for token counts. This provider
maps those into the OpenAI ``choices``/``usage`` shape.
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
from rezunate_llm_sdk.providers.base import BaseProvider, StreamState
from rezunate_llm_sdk.providers.endpoints import LLAMA_BASE_URL, LLAMA_CHAT_ENDPOINT, get_url
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
        finish_reason = FINISH_REASON_MAP.get(completion.stop_reason or "stop", FinishReason.STOP)

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

    # ---- streaming hooks (driven by BaseProvider.stream) ----------------
    def _build_stream_request(
        self, request: ChatCompletionRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        body = self.transform_request(request).model_dump(exclude_none=True)
        body["stream"] = True
        return get_url(self.base_url, self.get_endpoint()), body, self.get_headers()

    def _translate_frame(
        self, event: str, data: str, state: StreamState
    ) -> ChatCompletionChunk | None:
        payload = self._parse_json_frame(data)
        if payload is None:
            return None

        # Meta wraps frames in {"event": {"event_type": "...", ...}}.
        # If the SSE event name is set we trust it; otherwise look inside.
        inner = payload.get("event") if isinstance(payload.get("event"), dict) else payload
        event_type = event or inner.get("event_type") or "progress"

        if event_type == "start":
            state.id = payload.get("id") or inner.get("id") or state.id
            state.model = inner.get("model", state.model)
            return self._make_chunk(state, delta=ChoiceDelta(role=Role.ASSISTANT, content=""))

        if event_type == "progress":
            delta_block = (
                inner.get("delta") or inner.get("completion_message", {}).get("content") or {}
            )
            if text := (delta_block.get("text") or delta_block.get("content") or ""):
                return self._make_chunk(state, delta=ChoiceDelta(content=text))
            return None

        if event_type == "complete":
            stop_reason = inner.get("stop_reason")
            finish = FINISH_REASON_MAP.get(stop_reason, FinishReason.STOP) if stop_reason else None
            usage = self._usage_from_metrics(inner.get("metrics") or [])
            return self._make_chunk(state, finish_reason=finish, usage=usage)

        return None

    def _usage_from_metrics(self, metrics: list[dict]) -> Usage:
        m = {item.get("metric"): int(item.get("value", 0)) for item in metrics}
        prompt = m.get(constants.LLAMA_METRIC_PROMPT_TOKENS, 0)
        completion = m.get(constants.LLAMA_METRIC_COMPLETION_TOKENS, 0)
        total = m.get(constants.LLAMA_METRIC_TOTAL_TOKENS, prompt + completion)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )
