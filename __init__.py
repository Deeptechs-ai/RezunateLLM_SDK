"""
Python SDK Gateway
==================

A simple Python library for chat completions with multiple AI providers.
Uses OpenAI format as the standard for all requests and responses.

Supported Providers:
- OpenAI (gpt-4, gpt-3.5-turbo, etc.)
- Anthropic (claude-3-opus, claude-3-sonnet, etc.)
- Google (gemini-pro, gemini-1.5-pro, etc.)

Quick Start:
------------

    from python_sdk import chat_complete

    # Works with any provider - same format!
    response = chat_complete(
        provider="anthropic",
        api_key="your-api-key",
        model="claude-3-sonnet-20240229",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        max_tokens=100
    )

    print(response["choices"][0]["message"]["content"])

Using the Gateway class:
------------------------

    from python_sdk import Gateway

    gateway = Gateway(
        default_provider="openai",
        default_api_key="your-api-key"
    )

    response = gateway.chat_complete(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

from .gateway import chat_complete, get_available_providers, Gateway
from .providers import (
    PROVIDERS,
    get_provider,
    list_providers,
    BaseProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
)

__version__ = "0.1.0"

__all__ = [
    # Main functions
    "chat_complete",
    "get_available_providers",
    "Gateway",
    # Provider utilities
    "PROVIDERS",
    "get_provider",
    "list_providers",
    # Provider classes
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
]
