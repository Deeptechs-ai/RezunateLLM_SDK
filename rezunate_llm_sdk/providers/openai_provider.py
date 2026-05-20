"""OpenAI Provider — uses the official OpenAI SDK."""

from rezunate_llm_sdk.models import Provider
from rezunate_llm_sdk.providers.endpoints import OPENAI_BASE_URL
from rezunate_llm_sdk.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider."""

    BASE_URL = OPENAI_BASE_URL
    PROVIDER = Provider.OPENAI
