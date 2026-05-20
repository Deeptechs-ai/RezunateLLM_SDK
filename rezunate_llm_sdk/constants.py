"""Global constants for LLM Router."""

import os

# HTTP Headers
CONTENT_TYPE_HEADER = "Content-Type"
APPLICATION_JSON = "application/json"
AUTHORIZATION_HEADER = "Authorization"
API_KEY_HEADER = "x-api-key"
GOOGLE_API_KEY_HEADER = "x-goog-api-key"
ANTHROPIC_VERSION_HEADER = "anthropic-version"

# Retry Configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Router API
ROUTER_BASE_URL = os.getenv("ROUTER_BASE_URL", "https://rezunatellm.com")

# Llama (Meta) native API — well-known keys in the response ``metrics`` array
LLAMA_METRIC_PROMPT_TOKENS = "num_prompt_tokens"
LLAMA_METRIC_COMPLETION_TOKENS = "num_completion_tokens"
LLAMA_METRIC_TOTAL_TOKENS = "num_total_tokens"
