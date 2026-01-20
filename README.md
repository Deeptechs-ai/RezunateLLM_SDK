# LLM-Router

A unified Python SDK for chat completions across multiple LLM providers. Write once, use with OpenAI, Anthropic, Google Gemini, and more. All requests and responses follow the OpenAI format as the standard, making it easy to switch between providers without changing your code.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LLM-Router.git
cd LLM-Router

# Install dependencies
pip install requests
```

## Quick Start

### Using in Python REPL (Interactive Mode)

```bash
cd LLM-Router
python
```

```python
>>> from gateway import chat_complete
>>>
>>> # Using Anthropic
>>> response = chat_complete(
...     provider="anthropic",
...     api_key="sk-ant-your-key",
...     model="claude-sonnet-4-20250514",
...     messages=[{"role": "user", "content": "Hello!"}],
...     max_tokens=100
... )
>>> print(response["choices"][0]["message"]["content"])
Hello! How can I help you today?

>>> # Using Google Gemini
>>> response = chat_complete(
...     provider="google",
...     api_key="your-google-key",
...     model="gemini-2.0-flash",
...     messages=[{"role": "user", "content": "Hi!"}],
...     max_tokens=100
... )
>>> print(response["choices"][0]["message"]["content"])
```

### Using in a Python Script

```python
from gateway import chat_complete, get_available_providers

# List available providers
print(get_available_providers())  # ['openai', 'anthropic', 'google']

# Make a request
response = chat_complete(
    provider="anthropic",
    api_key="your-api-key",
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    temperature=0.7,
    max_tokens=200
)

# Get the response
print(response["choices"][0]["message"]["content"])
print(f"Tokens used: {response['usage']['total_tokens']}")
print(f"Provider: {response['provider']}")
```

### Using the Gateway Class

```python
from gateway import Gateway

# Create gateway with defaults
gw = Gateway(default_provider="anthropic", default_api_key="your-key")

# Make requests without repeating provider/key
response = gw.chat_complete(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| OpenAI | gpt-4, gpt-3.5-turbo, etc. | Supported |
| Anthropic | claude-opus-4, claude-sonnet-4, etc. | Supported |
| Google | gemini-2.0-flash, gemini-2.5-pro, etc. | Supported |

## Response Format

All providers return responses in OpenAI format:

```python
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "claude-sonnet-4-20250514",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you?"
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    },
    "provider": "anthropic"
}
```

## Project Structure

```
LLM-Router/
├── gateway.py              # Main entry point
├── types.py                # Type definitions (optional)
├── providers/
│   ├── __init__.py         # Package interface
│   ├── base.py             # Abstract base class
│   ├── factory.py          # Factory pattern implementation
│   ├── openai_provider.py  # OpenAI provider
│   ├── anthropic_provider.py # Anthropic provider
│   └── google_provider.py  # Google Gemini provider
```

## Adding a New Provider

1. Create `providers/your_provider.py` implementing `BaseProvider`
2. Add factory class in `providers/factory.py`
3. Register in `FACTORY_REGISTRY`

See existing providers for examples.
