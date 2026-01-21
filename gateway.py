"""
Gateway - Main Router.

This is the entry point for all chat completion requests.
Routes requests to the appropriate provider.
"""

from models import ChatCompletionRequest, ChatCompletionResponse, Message
from providers import get_provider, list_providers

__all__ = [
    "chat_complete",
    "get_available_providers",
    "Gateway",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Message",
]


def chat_complete(
    provider: str,
    api_key: str,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """
    Execute chat completion with any provider.

    All requests and responses use OpenAI format as the universal standard.

    Args:
        provider: Provider name ("openai", "anthropic", "google").
        api_key: API key for the provider.
        request: ChatCompletionRequest containing model, messages, and optional parameters.

    Returns:
        ChatCompletionResponse with id, object, created, model, choices, usage, and provider.

    Example:
        >>> request = ChatCompletionRequest(
        ...     model="claude-sonnet-4-20250514",
        ...     messages=[Message(role="user", content="Hello!")],
        ...     max_tokens=100,
        ... )
        >>> response = chat_complete(
        ...     provider="anthropic",
        ...     api_key="sk-ant-...",
        ...     request=request,
        ... )
        >>> print(response.choices[0].message.content)
    """
    request_dict = request.model_dump(exclude_none=True)

    provider_instance = get_provider(provider, api_key, model=request.model)
    response_dict = provider_instance.chat_complete(request_dict)

    return ChatCompletionResponse.model_validate(response_dict)


def get_available_providers() -> list[str]:
    """
    Get list of available providers.

    Returns:
        List of provider names.
    """
    return list_providers()


class Gateway:
    """
    Gateway class for object-oriented usage.

    Allows setting default provider and API key for convenience.

    Example:
        >>> gateway = Gateway(default_provider="openai", default_api_key="sk-...")
        >>> request = ChatCompletionRequest(
        ...     model="gpt-4",
        ...     messages=[Message(role="user", content="Hello")],
        ... )
        >>> response = gateway.chat_complete(request)
    """

    def __init__(
        self,
        default_provider: str | None = None,
        default_api_key: str | None = None,
    ) -> None:
        """
        Initialize Gateway.

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
        """
        Execute chat completion.

        Uses default provider and api_key if not specified.

        Args:
            request: ChatCompletionRequest containing model, messages, and optional parameters.
            provider: Provider name (uses default if not specified).
            api_key: API key (uses default if not specified).

        Returns:
            ChatCompletionResponse in OpenAI format.

        Raises:
            ValueError: If provider or api_key is not specified and no default is set.
        """
        resolved_provider = provider if provider is not None else self.default_provider
        resolved_api_key = api_key if api_key is not None else self.default_api_key

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
        """Get available providers."""
        return get_available_providers()
