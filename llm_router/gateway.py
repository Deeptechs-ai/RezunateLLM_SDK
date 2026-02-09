"""Gateway - Routes chat completion requests to providers."""

from llm_router.models import ChatCompletionRequest, ChatCompletionResponse
from llm_router.providers import get_provider, list_providers


def chat_complete(
    provider: str,
    api_key: str,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Execute chat completion with the specified provider.

    Args:
        provider: Provider name ("openai", "anthropic", "google").
        api_key: API key for the provider.
        request: Chat completion request with model, messages, and parameters.

    Returns:
        Chat completion response in OpenAI format.
    """
    provider_instance = get_provider(provider, api_key, model=request.model)
    response_dict = provider_instance.chat_complete(request.model_dump(exclude_none=True))
    return ChatCompletionResponse.model_validate(response_dict)


def get_available_providers() -> list[str]:
    """Get list of available providers.

    Returns:
        List of provider names.
    """
    return list_providers()


class Gateway:
    """Gateway class with default provider and API key support.

    Attributes:
        default_provider: Default provider to use when not specified.
        default_api_key: Default API key to use when not specified.
    """

    def __init__(
        self,
        default_provider: str | None = None,
        default_api_key: str | None = None,
    ) -> None:
        """Initialize Gateway.

        Args:
            default_provider: Default provider to use.
            default_api_key: Default API key to use.
        """
        self.default_provider = default_provider
        self.default_api_key = default_api_key

    def chat_complete(
        self,
        request: ChatCompletionRequest,
        provider: str | None = None,
        api_key: str | None = None,
    ) -> ChatCompletionResponse:
        """Execute chat completion.

        Args:
            request: Chat completion request with model, messages, and parameters.
            provider: Provider name (uses default if not specified).
            api_key: API key (uses default if not specified).

        Returns:
            Chat completion response in OpenAI format.

        Raises:
            ValueError: If provider or api_key is not specified and no default is set.
        """
        resolved_provider = provider or self.default_provider
        resolved_api_key = api_key or self.default_api_key

        if not resolved_provider:
            raise ValueError("Provider must be specified")
        if not resolved_api_key:
            raise ValueError("API key must be specified")

        return chat_complete(
            provider=resolved_provider,
            api_key=resolved_api_key,
            request=request,
        )

    @property
    def providers(self) -> list[str]:
        """Get available providers.

        Returns:
            List of provider names.
        """
        return get_available_providers()
