"""
Grok (xAI) Provider.
Uses the OpenAI SDK against xAI's OpenAI-compatible chat completions endpoint.
"""

from openai import OpenAI

from rezunate_llm_sdk.models import Provider
from rezunate_llm_sdk.providers.endpoints import GROK_BASE_URL, GROK_CHAT_ENDPOINT
from rezunate_llm_sdk.providers.openai_provider import OpenAIProvider


class GrokProvider(OpenAIProvider):
    """Grok (xAI) provider — uses xAI's OpenAI-compatible API."""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=GROK_BASE_URL,
            max_retries=self.max_retries,
        )

    @property
    def base_url(self) -> str:
        return GROK_BASE_URL

    @property
    def provider_name(self) -> Provider:
        return Provider.GROK

    def get_endpoint(self, model: str | None = None) -> str:
        return GROK_CHAT_ENDPOINT
