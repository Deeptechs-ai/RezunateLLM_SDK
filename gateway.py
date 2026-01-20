"""
Gateway - Main Router.

This is the entry point for all chat completion requests.
Routes requests to the appropriate provider.
"""

from typing import Any

from providers import get_provider, list_providers

__all__ = ["chat_complete", "get_available_providers", "Gateway"]


def chat_complete(
    provider: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute chat completion with any provider.

    All requests and responses use OpenAI format as the universal standard.

    Args:
        provider: Provider name ("openai", "anthropic", "google").
        api_key: API key for the provider.
        model: Model name (e.g., "gpt-4", "claude-sonnet-4-20250514", "gemini-2.0-flash").
        messages: List of messages in OpenAI format.
            Example: [{"role": "user", "content": "Hello"}]
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        **kwargs: Additional provider-specific parameters.

    Returns:
        Response dictionary in OpenAI format containing id, object, created,
        model, choices, usage, and provider fields.

    Example:
        >>> response = chat_complete(
        ...     provider="anthropic",
        ...     api_key="sk-ant-...",
        ...     model="claude-sonnet-4-20250514",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100,
        ... )
        >>> print(response["choices"][0]["message"]["content"])
    """
    # Build request in OpenAI format
    request = {
        "model": model,
        "messages": messages,
    }

    # Add optional parameters if provided
    if temperature is not None:
        request["temperature"] = temperature

    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    # Add any additional kwargs
    request.update(kwargs)

    # Get provider instance and execute
    provider_instance = get_provider(provider, api_key, model=model)
    response = provider_instance.chat_complete(request)

    return response


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
        ...     messages=[{"role": "user", "content": "Hello"}],
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
        messages: list[dict[str, str]],
        model: str,
        provider: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute chat completion.

        Uses default provider and api_key if not specified.

        Args:
            messages: List of messages in OpenAI format.
            model: Model name to use.
            provider: Provider name (uses default if not specified).
            api_key: API key (uses default if not specified).
            **kwargs: Additional provider-specific parameters.

        Returns:
            Response dictionary in OpenAI format.

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
            provider=provider, api_key=api_key, model=model, messages=messages, **kwargs
        )

    @property
    def providers(self) -> list[str]:
        """Get available providers."""
        return get_available_providers()
