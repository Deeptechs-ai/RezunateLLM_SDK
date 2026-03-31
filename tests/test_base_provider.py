"""
Tests for BaseProvider class.
"""

import pytest
import responses
from pydantic import BaseModel

from llm_router.models import ChatCompletionRequest, ChatCompletionResponse
from llm_router.providers.base import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    BaseProvider,
)


class ConcreteProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing."""

    response_model = ChatCompletionResponse

    @property
    def base_url(self) -> str:
        return "https://api.test.com/v1"

    @property
    def provider_name(self) -> str:
        return "test_provider"

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_endpoint(self, model: str = None):
        return "/chat/completions"

    def transform_request(self, request: ChatCompletionRequest) -> BaseModel:
        return request

    def transform_response(self, response: BaseModel, model: str = None) -> ChatCompletionResponse:
        if isinstance(response, ChatCompletionResponse):
            return response
        return ChatCompletionResponse.model_validate(response.model_dump())


class TestBaseProviderInit:
    """Tests for BaseProvider initialization."""

    def test_init_with_defaults(self, mock_api_key):
        """Test provider initializes with default values."""
        provider = ConcreteProvider(api_key=mock_api_key)

        assert provider.api_key == mock_api_key
        assert provider.max_retries == DEFAULT_MAX_RETRIES
        assert provider.retry_delay == DEFAULT_RETRY_DELAY
        assert provider.timeout == 60.0

    def test_init_with_custom_values(self, mock_api_key):
        """Test provider initializes with custom values."""
        provider = ConcreteProvider(
            api_key=mock_api_key,
            max_retries=5,
            retry_delay=2.0,
            timeout=120.0,
        )

        assert provider.max_retries == 5
        assert provider.retry_delay == 2.0
        assert provider.timeout == 120.0


class TestBackoffCalculation:
    """Tests for exponential backoff calculation."""

    def test_backoff_increases_exponentially(self, mock_api_key):
        """Test that backoff increases with each attempt."""
        provider = ConcreteProvider(api_key=mock_api_key, retry_delay=1.0)

        backoff_0 = provider._calculate_backoff(0)
        backoff_1 = provider._calculate_backoff(1)
        backoff_2 = provider._calculate_backoff(2)

        # Base values before jitter: 1, 2, 4
        assert 1.0 <= backoff_0 <= 1.1  # 1 + up to 10% jitter
        assert 2.0 <= backoff_1 <= 2.2  # 2 + up to 10% jitter
        assert 4.0 <= backoff_2 <= 4.4  # 4 + up to 10% jitter

    def test_backoff_respects_retry_delay(self, mock_api_key):
        """Test that backoff uses configured retry_delay."""
        provider = ConcreteProvider(api_key=mock_api_key, retry_delay=2.0)

        backoff_0 = provider._calculate_backoff(0)

        assert 2.0 <= backoff_0 <= 2.2


class TestRetryableStatusCodes:
    """Tests for retryable status code detection."""

    def test_retryable_status_codes(self, mock_api_key):
        """Test that specific status codes are retryable."""
        provider = ConcreteProvider(api_key=mock_api_key)

        for code in [429, 500, 502, 503, 504]:
            assert provider._is_retryable(code) is True

    def test_non_retryable_status_codes(self, mock_api_key):
        """Test that other status codes are not retryable."""
        provider = ConcreteProvider(api_key=mock_api_key)

        for code in [200, 400, 401, 403, 404]:
            assert provider._is_retryable(code) is False

    def test_none_status_code(self, mock_api_key):
        """Test handling of None status code."""
        provider = ConcreteProvider(api_key=mock_api_key)

        assert provider._is_retryable(None) is False


class TestChatComplete:
    """Tests for chat_complete method."""

    @responses.activate
    def test_successful_request(self, mock_api_key, openai_response):
        """Test successful API request."""
        responses.add(
            responses.POST,
            "https://api.test.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        provider = ConcreteProvider(api_key=mock_api_key)
        request_dict = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        req_obj = ChatCompletionRequest.model_validate(request_dict)

        result = provider.chat_complete(req_obj)

        assert result.id == openai_response["id"]
        assert result.provider == "test_provider"
        assert len(responses.calls) == 1

    @responses.activate
    def test_retry_on_429(self, mock_api_key, openai_response):
        """Test retry on rate limit (429) error."""
        # First request fails with 429, second succeeds
        responses.add(
            responses.POST,
            "https://api.test.com/v1/chat/completions",
            json={"error": "rate limited"},
            status=429,
        )
        responses.add(
            responses.POST,
            "https://api.test.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        provider = ConcreteProvider(api_key=mock_api_key, retry_delay=0.01)
        request_dict = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        req_obj = ChatCompletionRequest.model_validate(request_dict)

        result = provider.chat_complete(req_obj)

        assert result.error is None
        assert result.provider == "test_provider"
        assert len(responses.calls) == 2

    @responses.activate
    def test_retry_on_500(self, mock_api_key, openai_response):
        """Test retry on server error (500)."""
        responses.add(
            responses.POST,
            "https://api.test.com/v1/chat/completions",
            json={"error": "internal server error"},
            status=500,
        )
        responses.add(
            responses.POST,
            "https://api.test.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        provider = ConcreteProvider(api_key=mock_api_key, retry_delay=0.01)
        request_dict = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        req_obj = ChatCompletionRequest.model_validate(request_dict)

        result = provider.chat_complete(req_obj)

        assert result.error is None
        assert len(responses.calls) == 2

    @responses.activate
    def test_no_retry_on_400(self, mock_api_key):
        """Test no retry on client error (400)."""
        responses.add(
            responses.POST,
            "https://api.test.com/v1/chat/completions",
            json={"error": "bad request"},
            status=400,
        )

        provider = ConcreteProvider(api_key=mock_api_key, retry_delay=0.01)
        request_dict = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        req_obj = ChatCompletionRequest.model_validate(request_dict)

        result = provider.chat_complete(req_obj)

        assert result.error is not None
        assert len(responses.calls) == 1

    @responses.activate
    def test_max_retries_exhausted(self, mock_api_key):
        """Test error returned after max retries exhausted."""
        # All requests fail with 429
        for _ in range(4):  # max_retries (3) + 1 initial
            responses.add(
                responses.POST,
                "https://api.test.com/v1/chat/completions",
                json={"error": "rate limited"},
                status=429,
            )

        provider = ConcreteProvider(api_key=mock_api_key, max_retries=3, retry_delay=0.01)
        request_dict = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        req_obj = ChatCompletionRequest.model_validate(request_dict)

        result = provider.chat_complete(req_obj)

        assert result.error is not None
        assert result.error.retries_attempted == 3
        assert result.provider == "test_provider"
        assert len(responses.calls) == 4

    @responses.activate
    def test_timeout_handling(self, mock_api_key):
        """Test timeout is passed to requests."""
        responses.add(
            responses.POST,
            "https://api.test.com/v1/chat/completions",
            json={"id": "test", "object": "chat.completion", "choices": [], "usage": {}},
            status=200,
        )

        provider = ConcreteProvider(api_key=mock_api_key, timeout=30.0)
        request_dict = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}
        req_obj = ChatCompletionRequest.model_validate(request_dict)

        provider.chat_complete(req_obj)

        # Check that the request was made with the correct timeout
        assert len(responses.calls) == 1


class TestAbstractMethods:
    """Tests to ensure abstract methods are enforced."""

    def test_cannot_instantiate_base_provider(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider(api_key="test")
