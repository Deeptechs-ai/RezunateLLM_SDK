"""
Gateway - Main Router.

This is the entry point for all chat completion requests.
Routes requests to the appropriate provider.
"""

from models import ChatCompletionRequest, ChatCompletionResponse, Message
from providers import get_provider, list_providers

__all__ = ["chat_complete", "get_available_providers", "Gateway"]


def chat_complete(
    provider: str,
    api_key: str,
    model: str,
    messages: list[Message],
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    stop: str | list[str] | None = None,
    n: int | None = None,
    stream: bool | None = None,
    user: str | None = None,
) -> ChatCompletionResponse:
    """
    Execute chat completion with any provider.

    All requests and responses use OpenAI format as the universal standard.

    Args:
        provider: Provider name ("openai", "anthropic", "google").
        api_key: API key for the provider.
        model: Model name (e.g., "gpt-4", "claude-sonnet-4-20250514", "gemini-2.0-flash").
        messages: List of Message objects.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        frequency_penalty: Frequency penalty (-2.0 to 2.0).
        presence_penalty: Presence penalty (-2.0 to 2.0).
        stop: Stop sequences.
        n: Number of completions to generate.
        stream: Whether to stream the response.
        user: User identifier.

    Returns:
        ChatCompletionResponse with id, object, created, model, choices, usage, and provider.

    Example:
        >>> response = chat_complete(
        ...     provider="anthropic",
        ...     api_key="sk-ant-...",
        ...     model="claude-sonnet-4-20250514",
        ...     messages=[Message(role="user", content="Hello!")],
        ...     max_tokens=100,
        ... )
        >>> print(response.choices[0].message.content)
    """
    # Build request using Pydantic model
    request = ChatCompletionRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        stream=stream,
        user=user,
    )

    # Convert to dict for provider (excluding None values)
    request_dict = request.model_dump(exclude_none=True)

    # Get provider instance and execute
    provider_instance = get_provider(provider, api_key, model=model)
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
        >>> response = gateway.chat_complete(
        ...     model="gpt-4",
        ...     messages=[Message(role="user", content="Hello")],
        ... )
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
        messages: list[Message],
        model: str,
        provider: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        n: int | None = None,
        stream: bool | None = None,
        user: str | None = None,
    ) -> ChatCompletionResponse:
        """
        Execute chat completion.

        Uses default provider and api_key if not specified.

        Args:
            messages: List of Message objects.
            model: Model name to use.
            provider: Provider name (uses default if not specified).
            api_key: API key (uses default if not specified).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Frequency penalty (-2.0 to 2.0).
            presence_penalty: Presence penalty (-2.0 to 2.0).
            stop: Stop sequences.
            n: Number of completions to generate.
            stream: Whether to stream the response.
            user: User identifier.

        Returns:
            ChatCompletionResponse in OpenAI format.

        Raises:
            ValueError: If provider or api_key is not specified and no default is set.
        """
        provider = provider or self.default_provider
        api_key = api_key or self.default_api_key

        if not provider:
            raise ValueError("Provider must be specified")
        if not api_key:
            raise ValueError("API key must be specified")

        return chat_complete(
            provider=provider,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n,
            stream=stream,
            user=user,
        )

    @property
    def providers(self) -> list[str]:
        """Get available providers."""
        return get_available_providers()
