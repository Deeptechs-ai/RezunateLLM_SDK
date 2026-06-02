# Rezunate LLM SDK

Unified Python SDK for chat completions, prompt management, and PII guardrails across OpenAI, Anthropic, and Google Gemini. All requests and responses use the OpenAI format, regardless of provider.

Get an API key and learn more at [rezunatellm.com](https://rezunatellm.com).

## What's Included

| Feature | Cost | Requires |
|---|---|---|
| Unified chat completions across OpenAI, Anthropic, Google Gemini | **Free** | Your own provider API key |
| Local regex guardrails (block / flag PII in inputs and outputs) | **Free** | Nothing — runs client-side |
| Server-side PII detection | Hosted | Rezunate API key |
| Prompt management with versioning  | Hosted | Rezunate API key |

You only need a Rezunate API key for the hosted features.

## Installation

```bash
pip install rezunate-llm-sdk
```

## Quickstart

```python
from rezunate_llm_sdk import ChatCompletionRequest, Gateway, Message

gateway = Gateway(
    default_provider="anthropic",
    default_api_key="your-provider-api-key",
)

response = gateway.chat_complete(
    ChatCompletionRequest(
        model="claude-sonnet-4-5",
        messages=[Message(role="user", content="Hello!")],
        max_tokens=100,
    )
)

print(response.choices[0].message.content)
```

The stateless form is also available when you don't want a long-lived gateway:

```python
from rezunate_llm_sdk import ChatCompletionRequest, Message, chat_complete

response = chat_complete(
    provider="openai",
    api_key="your-openai-api-key",
    request=ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="Hello!")],
    ),
)
```

Responses always come back in OpenAI format (`response.choices[0].message.content`, `response.usage.total_tokens`, etc.) — even for Anthropic and Google.

## Providers

The SDK ships with a factory that creates provider instances on demand. You bring your own API key per provider — Rezunate doesn't proxy or charge for these calls.

```python
from rezunate_llm_sdk import get_available_providers, get_provider

# List the providers the SDK supports
print(get_available_providers())
# [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE]

# Build a provider instance directly (skips the Gateway/chat_complete facade)
provider = get_provider("anthropic", api_key="your-anthropic-key")
response = provider.chat_complete(
    ChatCompletionRequest(
        model="claude-sonnet-4-5",
        messages=[Message(role="user", content="Hi!")],
    )
)
```

You can also switch providers at call time on a single `Gateway`:

```python
gateway = Gateway()

gateway.chat_complete(req, provider="openai",    api_key=openai_key)
gateway.chat_complete(req, provider="anthropic", api_key=anthropic_key)
gateway.chat_complete(req, provider="google",    api_key=google_key)
```

### Request Parameters

`ChatCompletionRequest` follows the OpenAI schema and supports the common knobs:

```python
ChatCompletionRequest(
    model="gpt-4o",
    messages=[
        Message(role="system",    content="You are concise."),
        Message(role="user",      content="Summarize the last commit."),
    ],
    temperature=0.2,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["END"],
)
```

Provider-specific arguments (e.g. Google `top_k`, `safety_settings`, `tools`) are accepted as extra fields and forwarded by each provider's transformer.



## Local Regex Guardrails (Free)

Define regex patterns in YAML to **block**, **flag**, or **redact** sensitive content (PII or anything custom) in both user inputs and model outputs. Everything is free and runs client-side — no RezunateLLM account needed, no data leaves your machine.

Create a config file:

```yaml
# guardrails.yaml
guardrails:
  - name: block-ssn
    pattern: '\b\d{3}-\d{2}-\d{4}\b'
    description: "Block Social Security Numbers"
    action: block

  # Redact PII instead of blocking the whole request/response.
  - name: redact-email
    pattern: '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    description: "Redact email addresses"
    action: redact
    replacement: "[EMAIL]"

  - name: flag-credit-card
    pattern: '\b(?:\d[ -]*?){13,16}\b'
    description: "Flag potential credit card numbers"
    action: flag
```

Each rule needs:
- `name` — identifier
- `pattern` — regex to match
- `description` — human-readable description
- `action` — one of:
  - `block` — raises `GuardrailsError`, stopping the request (input) or response (output)
  - `flag` — records a violation and logs it, but lets the content through unchanged
  - `redact` — replaces every match with `replacement` (defaults to `[REDACTED]`) in both the prompt sent to the provider and the model's response
- `replacement` — text substituted for each match when `action: redact` (optional; defaults to `[REDACTED]`)

Use it in two ways:

**Pass it explicitly to a Gateway:**

```python
from rezunate_llm_sdk import Gateway, load_guardrails

config = load_guardrails("guardrails.yaml")
gateway = Gateway(
    default_provider="anthropic",
    default_api_key="your-anthropic-key",
    guardrails_config=config,
)

# Both input and output are checked on every chat_complete call.
gateway.chat_complete(request)
```

**Or enable automatic loading via env var:**

```bash
export GUARDRAILS_FILE_PATH=path/to/guardrails.yaml
```

When set, the SDK loads the file once and applies the rules on every `chat_complete` call without explicit wiring.

You can also call the checker directly. `check_guardrails` returns a `(redacted_text, violations)` tuple and raises `GuardrailsError` on a `block` rule:

```python
from rezunate_llm_sdk import GuardrailsError, check_guardrails, load_guardrails
from rezunate_llm_sdk.models import GuardrailDirection

config = load_guardrails("guardrails.yaml")  # with a redact rule for emails
try:
    redacted, violations = check_guardrails(
        "Email me at alex@example.com", config, GuardrailDirection.OUTPUT
    )
    print(redacted)  # -> "Email me at [EMAIL]"  (redact rules applied)
    # 'flag' rules show up in `violations`; 'block' rules raise GuardrailsError
except GuardrailsError as e:
    print(f"Blocked by rule '{e.rule_name}' on {e.direction.name}")
```

## Hosted Features

These call the Rezunate API and require `REZUNATE_LLM_API_KEY` (passed to `Gateway` or set in the env).

### Server-side PII Detection (Premium Feature)

Detect PII entities using the workspace guardrail config managed in RezunateLLM. Returns model-detected entities, with labels, scores, and offsets.

```python
gateway = Gateway(REZUNATE_LLM_API_KEY="your-rezunate-api-key")

result = gateway.guardrails.scan("My SSN is 123-45-6789 and my email is alex@example.com.")

for entity in result.entities:
    print(f"{entity.label}: {entity.text!r} (score={entity.score:.2f})")

print("action:",  result.action)    # action taken per workspace config
print("blocked:", result.blocked)   # whether the request was blocked
print("text:",    result.text)      # processed text (e.g. with PII redacted)
```

### Prompt Management

Fetch and render prompts stored on Rezunate LLM. Pin to a specific version, or omit `version` to use the current one. Variables are interpolated into the template before returning.

```python
gateway = Gateway(
    default_provider="anthropic",
    default_api_key="your-anthropic-api-key",
    REZUNATE_LLM_API_KEY="your-rezunate-api-key",
)

system_prompt = gateway.get_prompt(
    slug_id="customer-support-greeting",
    variables={"name": "Alex", "tier": "premium"},
    version=3,  # optional — defaults to current version
)

response = gateway.chat_complete(
    ChatCompletionRequest(
        model="claude-sonnet-4-5",
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user",   content="Where's my order?"),
        ],
    )
)
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `REZUNATE_LLM_API_KEY` | Rezunate API key — required for hosted features (prompts, server-side scan) |
| `OPENAI_API_KEY` | OpenAI provider key — used by your application code |
| `ANTHROPIC_AI_API_KEY` | Anthropic provider key — used by your application code |
| `GOOGLE_API_KEY` | Google Gemini provider key — used by your application code |
| `GUARDRAILS_FILE_PATH` | Optional path to a local guardrails YAML config; loaded automatically when set |
