"""
DeepSeek Provider.
Uses the OpenAI SDK against DeepSeek's OpenAI-compatible chat completions endpoint.
"""

from typing import Any

from openai import OpenAI

from rezunate_llm_sdk.models import FINISH_REASON_MAP, ChatCompletionResponse, Provider
from rezunate_llm_sdk.providers.endpoints import DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_ENDPOINT
from rezunate_llm_sdk.providers.openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider — uses DeepSeek's OpenAI-compatible API."""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=DEEPSEEK_BASE_URL,
            max_retries=self.max_retries,
        )

    @property
    def base_url(self) -> str:
        return DEEPSEEK_BASE_URL

    @property
    def provider_name(self) -> Provider:
        return Provider.DEEPSEEK

    def get_endpoint(self, model: str | None = None) -> str:
        return DEEPSEEK_CHAT_ENDPOINT

    def transform_response(
        self, response: Any, model: str | None = None
    ) -> ChatCompletionResponse:
        """Normalize DeepSeek-specific finish_reason values before OpenAI validation.

        ``deepseek-reasoner`` can emit ``insufficient_system_resource`` under
        load — not a value in OpenAI's FinishReason enum, so plain pydantic
        validation would drop the choice. We rewrite it via FINISH_REASON_MAP
        (which maps it to STOP) before delegating to the standard parser.
        """
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif isinstance(response, dict):
            data = response
        else:
            return response

        for choice in data.get("choices") or []:
            fr = choice.get("finish_reason")
            if fr and fr in FINISH_REASON_MAP:
                choice["finish_reason"] = FINISH_REASON_MAP[fr].value

        return ChatCompletionResponse.model_validate(data)
