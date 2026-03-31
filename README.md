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

## Terminal Usage

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

## Guardrails

### Regex Guardrails

Regex guardrails let you block or flag sensitive content (SSNs, emails, credit cards, etc.) in both input and output messages using pattern matching.

#### Setup

1. Create a YAML config file (see `guardrails.example.yaml`):

Each rule requires:
- `name`: Identifier for the rule
- `pattern`: Regex pattern to match
- `description`: Human-readable description
- `action`: `block` (raises `GuardrailsError`) or `flag` (logs a warning but allows the request)

2. Set the environment variable to enable automatic loading:

```bash
GUARDRAILS_FILE_PATH=path-to-guardrails-config-yaml-file
```

#### Usage

Guardrails are applied automatically when `GUARDRAILS_FILE_PATH` is set. Both user input and LLM output are checked against all rules.
