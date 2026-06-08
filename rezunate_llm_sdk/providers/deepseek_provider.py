"""DeepSeek Provider — uses DeepSeek's OpenAI-compatible API."""

from typing import Any

from rezunate_llm_sdk.models import FINISH_REASON_MAP, ChatCompletionResponse, Provider
from rezunate_llm_sdk.providers.endpoints import DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_ENDPOINT
from rezunate_llm_sdk.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider."""

    BASE_URL = DEEPSEEK_BASE_URL
    PROVIDER = Provider.DEEPSEEK
    CHAT_ENDPOINT = DEEPSEEK_CHAT_ENDPOINT

    def transform_response(self, response: Any, model: str | None = None) -> ChatCompletionResponse:
        """Normalize DeepSeek-specific finish_reason values before validation.

        ``deepseek-reasoner`` can emit ``insufficient_system_resource`` under
        load — not a value in OpenAI's FinishReason enum, so plain validation
        would drop the choice. Rewrite it via FINISH_REASON_MAP first.
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
