"""
Gateway - Main Router.
This is the entry point for all chat completion requests.
Routes requests to the appropriate provider.
"""

from typing import Dict, Any, List, Optional
from providers import get_provider, list_providers


def chat_complete(
    provider: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute chat completion with any provider.

    All requests use OpenAI format.
    All responses return OpenAI format.

    Args:
        provider: Provider name ("openai", "anthropic", "google")
        api_key: API key for the provider
        model: Model name (e.g., "gpt-4", "claude-3-sonnet-20240229", "gemini-pro")
        messages: List of messages in OpenAI format
                  [{"role": "user", "content": "Hello"}]
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional provider-specific parameters

    Returns:
        OpenAI format response:
        {
            "id": "chatcmpl-...",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "...",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "..."},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            "provider": "openai"
        }

    Example:
        >>> response = chat_complete(
        ...     provider="anthropic",
        ...     api_key="sk-ant-...",
        ...     model="claude-3-sonnet-20240229",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     max_tokens=100
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


def get_available_providers() -> List[str]:
    """
    Get list of available providers.

    Returns:
        List of provider names
    """
    return list_providers()


# Convenience class for object-oriented usage
class Gateway:
    """
    Gateway class for object-oriented usage.

    Example:
        >>> gateway = Gateway()
        >>> response = gateway.chat_complete(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """

    def __init__(self, default_provider: Optional[str] = None, default_api_key: Optional[str] = None):
        """
        Initialize Gateway.

        Args:
            default_provider: Default provider to use
            default_api_key: Default API key to use
        """
        self.default_provider = default_provider
        self.default_api_key = default_api_key

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute chat completion.

        Uses default provider/api_key if not specified.
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
            **kwargs
        )

    @property
    def providers(self) -> List[str]:
        """Get available providers."""
        return get_available_providers()
