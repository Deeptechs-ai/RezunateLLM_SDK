from collections.abc import Iterator
from typing import Any, ClassVar

from openai import OpenAI

import rezunate_llm_sdk.constants as constants
from rezunate_llm_sdk.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Provider,
)
from rezunate_llm_sdk.providers.base import BaseProvider
from rezunate_llm_sdk.providers.endpoints import OPENAI_CHAT_ENDPOINT


class OpenAICompatibleProvider(BaseProvider):
    """Provider base class for OpenAI-compatible REST APIs.

    All Chat Completions traffic flows through the official ``openai`` SDK
    with a custom ``base_url``. Streaming is delegated to the SDK and
    yielded as :class:`~rezunate_llm_sdk.models.ChatCompletionChunk`.
    """

    BASE_URL: ClassVar[str]
    PROVIDER: ClassVar[Provider]
    CHAT_ENDPOINT: ClassVar[str] = OPENAI_CHAT_ENDPOINT

    response_model = ChatCompletionResponse

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
            max_retries=self.max_retries,
        )

    @property
    def base_url(self) -> str:
        return self.BASE_URL

    @property
    def provider_name(self) -> Provider:
        return self.PROVIDER

    def get_headers(self) -> dict[str, str]:
        """Default Bearer-token auth. Override if a provider uses something else."""
        return {
            constants.AUTHORIZATION_HEADER: f"Bearer {self.api_key}",
            constants.CONTENT_TYPE_HEADER: constants.APPLICATION_JSON,
        }

    def get_endpoint(self, model: str | None = None) -> str:
        return self.CHAT_ENDPOINT

    def transform_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Pass-through — request is already in OpenAI format."""
        return request

    def _execute_request(
        self, provider_request: ChatCompletionRequest, model: str | None = None
    ) -> Any:
        params = provider_request.model_dump(exclude_none=True)
        return self.client.chat.completions.create(**params, timeout=self.timeout)

    def transform_response(self, response: Any, model: str | None = None) -> ChatCompletionResponse:
        """Convert SDK response to our internal model.

        Providers that need to normalize quirky finish_reason values
        (e.g. DeepSeek) override this with a small pre-validation step.
        """
        if hasattr(response, "model_dump"):
            return ChatCompletionResponse.model_validate(response.model_dump())
        return response

    # ---- streaming (uses the SDK, not the BaseProvider SSE driver) ------

    def _stream_params(self, request: ChatCompletionRequest) -> dict[str, Any]:
        params = request.model_dump(exclude_none=True)
        params["stream"] = True
        return params

    def stream(self, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
        """Stream chunks via the OpenAI SDK's sync iterator."""
        try:
            sdk_stream = self.client.chat.completions.create(
                **self._stream_params(request), timeout=self.timeout
            )
            for sdk_chunk in sdk_stream:
                chunk = ChatCompletionChunk.model_validate(sdk_chunk.model_dump())
                chunk.provider = self.provider_name
                yield chunk
        except Exception as e:
            yield self._error_chunk(e, request.model)
