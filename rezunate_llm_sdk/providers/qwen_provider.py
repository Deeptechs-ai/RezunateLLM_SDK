"""
Qwen (Alibaba) Provider — native DashScope API.

Transforms OpenAI format <-> Alibaba DashScope native text-generation format.
Uses the international Singapore region by default.
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
    Role,
    Usage,
)
from rezunate_llm_sdk.providers.base import BaseProvider
from rezunate_llm_sdk.providers.endpoints import QWEN_BASE_URL, QWEN_GENERATION_ENDPOINT
from rezunate_llm_sdk.providers.qwen_models import (
    QwenInput,
    QwenMessage,
    QwenParameters,
    QwenRequest,
    QwenResponse,
)


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
