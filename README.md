# LLM-Router

Unified Python SDK for chat completions across OpenAI, Anthropic, and Google Gemini. All requests/responses use OpenAI format.

## Clone the Project 
```bash
git clone https://github.com/Deeptechs-ai/LLM-Router
```

## Installation

```bash
uv sync
```

## Usage

```python
from gateway import chat_complete, ChatCompletionRequest, Message

request = ChatCompletionRequest(
    model="claude-sonnet-4-20250514",
    messages=[Message(role="user", content="Hello!")],
    max_tokens=100,
)

response = chat_complete(
    provider="anthropic",  # or "openai", "google"
    api_key="your-api-key",
    request=request,
)

print(response.choices[0].message.content)
```

## Supported Providers

| Provider | Example Models |
|----------|----------------|
| OpenAI | gpt-4, gpt-3.5-turbo |
| Anthropic | claude-opus-4, claude-sonnet-4 |
| Google | gemini-2.0-flash, gemini-2.5-pro |
