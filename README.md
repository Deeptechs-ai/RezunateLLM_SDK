# Rezunate LLM SDK

Unified Python SDK for chat completions, prompt management, and PII guardrails across OpenAI, Anthropic, Google Gemini, Grok (xAI), DeepSeek, Llama (Meta), and Qwen (Alibaba). All requests and responses use the OpenAI format, regardless of provider — streaming included.

Get an API key and learn more at [rezunatellm.com](https://rezunatellm.com).

## What's Included

| Feature | Cost | Requires |
|---|---|---|
| Unified chat completions across seven providers, streaming or not | **Free** | Your own provider API key |
| Local regex guardrails (block / flag PII in inputs and outputs) | **Free** | Nothing — runs client-side |
| Server-side PII detection | Hosted | Rezunate API key |
| Prompt management with versioning  | Hosted | Rezunate API key |
| `rezunate-guard` — PII redaction for Claude Code, no code required | Hosted | Rezunate API key |

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

Responses always come back in OpenAI format (`response.choices[0].message.content`, `response.usage.total_tokens`, etc.) — whichever provider served the call.

## Providers

Seven providers ship with the SDK, all behind the same request and response shape:

| Provider | `provider=` | Wire protocol |
|---|---|---|
| OpenAI | `openai` | OpenAI Chat Completions |
| Anthropic | `anthropic` | Anthropic Messages |
| Google Gemini | `google` | Gemini `generateContent` |
| Grok (xAI) | `grok` | OpenAI-compatible |
| DeepSeek | `deepseek` | OpenAI-compatible |
| Llama (Meta) | `llama` | Meta native |
| Qwen (Alibaba) | `qwen` | DashScope native |

The SDK ships with a factory that creates provider instances on demand. You bring your own API key per provider — Rezunate doesn't proxy or charge for these calls.

```python
from rezunate_llm_sdk import get_available_providers, get_provider

# List the providers the SDK supports
print(get_available_providers())
# [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE,
#  Provider.GROK, Provider.LLAMA, Provider.DEEPSEEK, Provider.QWEN]

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

gateway.chat_complete(req, provider="openai", api_key=openai_key)
gateway.chat_complete(req, provider="anthropic", api_key=anthropic_key)
gateway.chat_complete(req, provider="google", api_key=google_key)
```

### Request Parameters

`ChatCompletionRequest` follows the OpenAI schema and supports the common knobs:

```python
ChatCompletionRequest(
    model="gpt-4o",
    messages=[
        Message(role="system", content="You are concise."),
        Message(role="user", content="Summarize the last commit."),
    ],
    temperature=0.2,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["END"],
)
```

Provider-specific arguments (e.g. Google `top_k`, `safety_settings`) are accepted as extra fields and forwarded by each provider's transformer.

## Streaming

Set `stream=True` and the call returns an iterator of `ChatCompletionChunk` objects instead of a single response. Every provider streams, and every one yields the same OpenAI chunk shape.

```python
req = ChatCompletionRequest(
    model="claude-sonnet-4-5",
    messages=[Message(role="user", content="Write a haiku about latency.")],
    max_tokens=200,
    stream=True,
)

for chunk in gateway.chat_complete(req):
    if chunk.error:
        print("stream failed:", chunk.error.message)
        break
    for choice in chunk.choices:
        if choice.delta.content:
            print(choice.delta.content, end="", flush=True)
```

`delta.role` is set on the first chunk only, `finish_reason` on the last. A provider or transport failure does not raise — the iterator yields a single chunk with `error` populated and stops, so check `chunk.error` in the loop.

Guardrails behave differently on a stream: local regex `block` rules still raise, but local `redact` rules only log (deltas are not rewritten). Hosted guardrails buffer output to a sentence boundary, scan, and emit redacted text — use `server_guardrails=True` when redacting streamed output matters.

## Tool Calling

Define tools in OpenAI format and the SDK translates them to each provider's native shape (Anthropic `tool_use`, Google `functionCall`) and normalizes the response back to OpenAI `tool_calls`.

Tool support by provider: OpenAI, Grok, and DeepSeek pass tools through natively; Anthropic and Google are translated both ways; Llama forwards `tools` to Meta's native field. Qwen's DashScope transformer does not carry tools — they are dropped on that provider.

```python
from rezunate_llm_sdk import ChatCompletionRequest, Gateway, Message

gateway = Gateway(default_provider="anthropic", default_api_key="your-anthropic-key")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

messages = [Message(role="user", content="What's the weather in Tokyo?")]

response = gateway.chat_complete(
    ChatCompletionRequest(
        model="claude-sonnet-4-5",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # "auto" | "required" | "none" | {"type": "function", "function": {"name": "get_weather"}}
        max_tokens=300,
    )
)

choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    call = choice.message.tool_calls[0]
    print(call.function.name)  # "get_weather"
    print(call.function.arguments)  # '{"city": "Tokyo"}'  (a JSON string)
```

Run the tool, then send the result back as a `tool` message to get the final answer:

```python
import json

args = json.loads(call.function.arguments)
result = f"Sunny, 22C in {args['city']}"  # your real tool goes here

messages += [
    Message(role="assistant", tool_calls=choice.message.tool_calls),
    Message(role="tool", content=result, tool_call_id=call.id),
    # For Google, also set name=call.function.name on the tool message.
]

final = gateway.chat_complete(
    ChatCompletionRequest(model="claude-sonnet-4-5", messages=messages, tools=tools, max_tokens=300)
)
print(final.choices[0].message.content)
```


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

print("action:", result.action)  # action taken per workspace config
print("blocked:", result.blocked)  # whether the request was blocked
print("text:", result.text)  # processed text (e.g. with PII redacted)
```

#### Automatic PII guardrails on input and output

Set `server_guardrails=True` and the Gateway scans every `chat_complete` call with the hosted PII service, applying the workspace config automatically:

- if the scan **blocks** the content → raises `ServerGuardrailsError`
- if the scan **redacts** it → the content is replaced with the server's redacted text (the prompt before it's sent, or the response after)
- detected entities are logged either way

```python
from rezunate_llm_sdk import Gateway, ServerGuardrailsError

gateway = Gateway(
    default_provider="openai",
    default_api_key="your-openai-key",
    server_guardrails=True,  # scan input AND output via hosted PII service
    REZUNATE_LLM_API_KEY="your-rezunate-api-key",
)

try:
    response = gateway.chat_complete(request)
    print(response.choices[0].message.content)  # PII redacted by the server
except ServerGuardrailsError as e:
    print(f"Blocked on {e.direction.name} — detected {[ent.label for ent in e.entities]}")
```

**Choosing which side to scan.** `True` scans both the prompt (input) and the model response (output). To scan only one side, pass a `ServerGuardrailsConfig` with `directions`:

```python
from rezunate_llm_sdk import Gateway, ServerGuardrailsConfig

gateway = Gateway(
    default_provider="openai",
    default_api_key="your-openai-key",
    server_guardrails=ServerGuardrailsConfig(directions=("output",)),  # output only
    REZUNATE_LLM_API_KEY="your-rezunate-api-key",
)
```

| `server_guardrails` value | Scans |
|---|---|
| `True` | input **and** output |
| `ServerGuardrailsConfig(directions=("input",))` | input only |
| `ServerGuardrailsConfig(directions=("output",))` | output only |
| `False` (default) | nothing |

You can also override it per call: `gateway.chat_complete(request, server_guardrails=...)`. This runs alongside any local regex guardrails.

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
            Message(role="user", content="Where's my order?"),
        ],
    )
)
```

## Claude Code PII Guard

`rezunate-guard` redacts PII from your files **before Claude Code sends them to the
model**. It needs no code — you point it at the folders holding personal data and it
works from then on.

```bash
cd your-project
rezunate-guard install     # register the hooks with Claude Code
rezunate-guard login       # save your Rezunate API key
rezunate-guard init        # write .rezunate-guard.yaml
rezunate-guard status      # confirm protection is actually on
```

Then list the folders to protect:

```yaml
# .rezunate-guard.yaml
scan:
  - clients
  - data/patients
  - /mnt/share/case-files    # absolute paths work too
```

Everything inside a listed folder is protected, at any depth. When Claude reads one of
those files, the personal data your workspace is configured to detect is replaced with
placeholders before the model sees them:

```
Ali Hassan emailed Sara Khan, and Sara Khan replied to Ali Hassan
[PERSON_NAME_a786] [PERSON_NAME_f54e] emailed [PERSON_NAME_3225] [PERSON_NAME_f4f5], and [PERSON_NAME_3225] [PERSON_NAME_f4f5] replied to [PERSON_NAME_a786] [PERSON_NAME_f54e]
```

The same value always gets the same placeholder, across files and sessions, so the
text still reads sensibly and Claude can tell the two people apart. It matches on the
value, not the person — `Ali Hassan` and `Hassan` are two different values and get two
different placeholders. Placeholders come from a key that never leaves your machine.

**Only the entity types enabled for your workspace are redacted.** If phone numbers are
off in your guardrail settings, a phone number in a protected file reaches the model in
the clear. Check what is enabled before relying on this.

| Command | What it does |
|---|---|
| `rezunate-guard install` | register the hooks in Claude Code's `settings.json` |
| `rezunate-guard login` | save your API key to `~/.rezunate/credentials` |
| `rezunate-guard init` | write a `.rezunate-guard.yaml` template here |
| `rezunate-guard uninstall` | remove them again, leaving other hooks alone |
| `rezunate-guard status` | report whether anything is actually being protected |
| `rezunate-guard check <paths>` | show whether given paths would be scanned, and why |

**Nothing is scanned unless you name it**, and each scanned file is a billable API call,
so list only what holds personal data. If the config cannot be read — a typo in the YAML,
a folder that isn't there — the guard protects nothing and writes the reason to
`~/.rezunate/guard.log`. `rezunate-guard status` reports the same thing and exits
non-zero, so check it after editing the config.

## Environment Variables

| Variable | Purpose |
|---|---|
| `REZUNATE_LLM_API_KEY` | Rezunate API key — required for hosted features (prompts, server-side scan) |
| `OPENAI_API_KEY` | OpenAI provider key — used by your application code |
| `ANTHROPIC_AI_API_KEY` | Anthropic provider key — used by your application code |
| `GOOGLE_API_KEY` | Google Gemini provider key — used by your application code |
| `GUARDRAILS_FILE_PATH` | Optional path to a local guardrails YAML config; loaded automatically when set |
| `REZUNATE_LLM_BASE_URL` | Rezunate API base URL (default `https://rezunatellm.com`) |
| `REZUNATE_HOME` | Where `rezunate-guard` keeps its key and log (default `~/.rezunate`) |
| `CLAUDE_CONFIG_DIR` | Where `rezunate-guard install` writes hook settings (default `~/.claude`) |
