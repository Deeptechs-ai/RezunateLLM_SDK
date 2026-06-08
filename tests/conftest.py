"""
Shared test fixtures for LLM Router tests.
"""

import pytest
import responses


@pytest.fixture
def mock_api_key():
    """Return a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_messages():
    """Return sample messages in OpenAI format."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def sample_messages_no_system():
    """Return sample messages without system message."""
    return [
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def sample_conversation():
    """Return a multi-turn conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What is 2+2?"},
    ]


@pytest.fixture
def openai_request(sample_messages):
    """Return a sample OpenAI format request."""
    return {
        "model": "gpt-4",
        "messages": sample_messages,
        "temperature": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def openai_response():
    """Return a sample OpenAI format response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def anthropic_response():
    """Return a sample Anthropic API response."""
    return {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello! How can I assist you today?",
            }
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
        },
    }


@pytest.fixture
def google_response():
    """Return a sample Google Gemini API response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello! How can I assist you today?"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
        },
    }


@pytest.fixture
def grok_response():
    """Return a sample xAI (Grok) chat completion response — OpenAI-shaped."""
    return {
        "id": "chatcmpl-grok-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "grok-3-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def deepseek_response():
    """Return a sample DeepSeek chat completion response — OpenAI-shaped."""
    return {
        "id": "chatcmpl-deepseek-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def qwen_response():
    """Return a sample DashScope (native Qwen) API response."""
    return {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?",
                    },
                }
            ]
        },
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
        "request_id": "req-test-123",
    }


@pytest.fixture
def llama_response():
    """Return a sample Meta Llama (native) API response."""
    return {
        "id": "msg-test-123",
        "completion_message": {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": "Hello! How can I assist you today?",
            },
            "stop_reason": "stop",
        },
        "metrics": [
            {"metric": "num_prompt_tokens", "value": 10, "unit": "tokens"},
            {"metric": "num_completion_tokens", "value": 20, "unit": "tokens"},
            {"metric": "num_total_tokens", "value": 30, "unit": "tokens"},
        ],
    }


@pytest.fixture
def mocked_responses():
    """Activate responses mock for HTTP requests."""
    with responses.RequestsMock() as rsps:
        yield rsps
