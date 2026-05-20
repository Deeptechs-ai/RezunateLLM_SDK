"""
Qwen (Alibaba) Provider — native DashScope API.

Transforms OpenAI format <-> Alibaba DashScope native text-generation format.
Uses the international Singapore region by default.
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
from rezunate_llm_sdk.providers.endpoints import QWEN_BASE_URL, QWEN_GENERATION_ENDPOINT, get_url
from rezunate_llm_sdk.providers.qwen_models import (
    QwenInput,
    QwenMessage,
    QwenParameters,
    QwenRequest,
    QwenResponse,
)

# DashScope opts into SSE streaming via this request header.
_DASHSCOPE_SSE_HEADER = "X-DashScope-SSE"


class QwenProvider(BaseProvider):
    """Qwen provider against Alibaba DashScope's native generation API."""

    response_model = QwenResponse

    @property
    def base_url(self) -> str:
        return QWEN_BASE_URL

    @property
    def provider_name(self) -> Provider:
        return Provider.QWEN

    def get_headers(self) -> dict[str, str]:
        return {
            constants.AUTHORIZATION_HEADER: f"Bearer {self.api_key}",
            constants.CONTENT_TYPE_HEADER: constants.APPLICATION_JSON,
        }

    def get_endpoint(self, model: str | None = None) -> str:
        return QWEN_GENERATION_ENDPOINT

    def transform_request(self, request: ChatCompletionRequest) -> QwenRequest:
        """Transform OpenAI format request to DashScope native format.

        Key differences:
          - OpenAI: flat ``messages`` array on root.
          - DashScope: messages live under ``input.messages``.
          - OpenAI: sampling params on root (temperature, top_p, max_tokens).
          - DashScope: sampling params live under ``parameters`` and
            ``result_format`` must be set to "message" for OpenAI-like
            response shape.
        """
        qwen_messages = [
            QwenMessage(role=msg.role.value, content=msg.content) for msg in request.messages
        ]

        stop = request.stop
        parameters = QwenParameters(
            result_format="message",
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=getattr(request, "top_k", None),
            max_tokens=request.max_tokens,
            stop=stop,
            seed=getattr(request, "seed", None),
            enable_search=getattr(request, "enable_search", None),
            repetition_penalty=getattr(request, "repetition_penalty", None),
        )

        return QwenRequest(
            model=request.model,
            input=QwenInput(messages=qwen_messages),
            parameters=parameters,
        )

    def transform_response(
        self, response: QwenResponse, model: str | None = None
    ) -> ChatCompletionResponse:
        """Transform DashScope native response to OpenAI format.

        DashScope returns either:
          - result_format="message": ``output.choices[].message`` (OpenAI-like)
          - result_format="text":    ``output.text`` + ``output.finish_reason``

        We always request "message" but handle both for robustness.
        """
        choices: list[Choice] = []
        if response.output.choices:
            for idx, choice in enumerate(response.output.choices):
                finish_reason = FINISH_REASON_MAP.get(
                    choice.finish_reason or "stop", FinishReason.STOP
                )
                choices.append(
                    Choice(
                        index=idx,
                        message=ResponseMessage(content=choice.message.content),
                        finish_reason=finish_reason,
                    )
                )
        elif response.output.text is not None:
            # Legacy result_format="text" path
            finish_reason = FINISH_REASON_MAP.get(
                response.output.finish_reason or "stop", FinishReason.STOP
            )
            choices.append(
                Choice(
                    index=0,
                    message=ResponseMessage(content=response.output.text),
                    finish_reason=finish_reason,
                )
            )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = response.usage.total_tokens or (input_tokens + output_tokens)

        return ChatCompletionResponse(
            id=response.request_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
            provider=self.provider_name,
        )

    # ---- streaming hooks (driven by BaseProvider.stream) ----------------

    def _build_stream_request(
        self, request: ChatCompletionRequest
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        body = self.transform_request(request).model_dump(exclude_none=True)
        # Ensure incremental_output is enabled so each frame is a delta.
        body.setdefault("parameters", {})["incremental_output"] = True

        headers = {**self.get_headers(), _DASHSCOPE_SSE_HEADER: "enable"}
        url = get_url(self.base_url, self.get_endpoint())
        return url, body, headers

    def _translate_frame(
        self, event: str, data: str, state: StreamState
    ) -> ChatCompletionChunk | None:
        payload = self._parse_json_frame(data)
        if payload is None:
            return None

        output = payload.get("output") or {}
        text = ""
        finish_reason_raw: str | None = None
        if choices_payload := (output.get("choices") or []):
            first = choices_payload[0]
            text = (first.get("message") or {}).get("content") or ""
            finish_reason_raw = first.get("finish_reason")
        elif output.get("text") is not None:
            text = output.get("text") or ""
            finish_reason_raw = output.get("finish_reason")

        # DashScope sends "null"/"" finish_reason while streaming; only the
        # final frame carries a real value.
        finish_reason = None
        if finish_reason_raw and finish_reason_raw != "null":
            finish_reason = FINISH_REASON_MAP.get(finish_reason_raw, FinishReason.STOP)

        usage = None
        if usage_payload := payload.get("usage"):
            input_tokens = usage_payload.get("input_tokens", 0)
            output_tokens = usage_payload.get("output_tokens", 0)
            total_tokens = usage_payload.get("total_tokens") or (input_tokens + output_tokens)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        delta = ChoiceDelta()
        if not state.role_sent:
            delta.role = Role.ASSISTANT
            state.role_sent = True
        if text:
            delta.content = text

        # Skip frames that carry neither content nor a terminal signal.
        if not text and finish_reason is None and usage is None and state.role_sent:
            return None

        # Prefer the per-frame request_id over the state's uuid placeholder.
        if request_id := payload.get("request_id"):
            state.id = request_id

        return self._make_chunk(state, delta=delta, finish_reason=finish_reason, usage=usage)
