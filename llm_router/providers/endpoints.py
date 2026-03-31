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
