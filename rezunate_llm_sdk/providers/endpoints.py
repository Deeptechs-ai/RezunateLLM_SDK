from urllib.parse import urljoin

# OpenAI
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_CHAT_ENDPOINT = "/chat/completions"

# Anthropic
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_MESSAGES_ENDPOINT = "/messages"
ANTHROPIC_DEFAULT_VERSION = "2023-06-01"

# Google (Gemini)
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GOOGLE_GENERATE_CONTENT_ENDPOINT = "/models/{model}:generateContent"
GOOGLE_STREAM_GENERATE_CONTENT_ENDPOINT = "/models/{model}:streamGenerateContent?alt=sse"

# Grok (xAI) — OpenAI-compatible API
GROK_BASE_URL = "https://api.x.ai/v1"
GROK_CHAT_ENDPOINT = "/chat/completions"

# Llama (Meta) — native API with Meta-specific schema
# (Use /compat/v1 instead if OpenAI-compatible shape is preferred.)
LLAMA_BASE_URL = "https://api.llama.com/v1"
LLAMA_CHAT_ENDPOINT = "/chat/completions"

# DeepSeek — OpenAI-compatible API
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_ENDPOINT = "/chat/completions"

# Qwen (Alibaba DashScope) — native API (Singapore international region)
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
QWEN_GENERATION_ENDPOINT = "/services/aigc/text-generation/generation"


def get_url(base_url: str, endpoint: str, model: str | None = None) -> str:
    """
    Robustly join base_url and endpoint.
    Handles trailing/leading slashes and string formatting for model name.
    """
    # Ensure base_url ends with a slash and endpoint does NOT start with one
    # This makes urljoin work predictably for path joining
    base = base_url if base_url.endswith("/") else f"{base_url}/"
    path = endpoint.lstrip("/")

    url = urljoin(base, path)

    if model:
        return url.format(model=model)
    return url
