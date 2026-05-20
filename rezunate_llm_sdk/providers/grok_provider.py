"""Grok (xAI) Provider — uses xAI's OpenAI-compatible API."""

from rezunate_llm_sdk.models import Provider
from rezunate_llm_sdk.providers.endpoints import GROK_BASE_URL, GROK_CHAT_ENDPOINT
from rezunate_llm_sdk.providers.openai_compatible import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    """Grok provider."""

    BASE_URL = GROK_BASE_URL
    PROVIDER = Provider.GROK
    CHAT_ENDPOINT = GROK_CHAT_ENDPOINT
