"""
Tests for OpenAI Provider.
"""

import responses

from llm_router.providers.openai_provider import OpenAIProvider


class TestOpenAIProviderProperties:
    """Tests for OpenAI provider properties."""

    def test_base_url(self, mock_api_key):
        """Test base URL is correct."""
        provider = OpenAIProvider(api_key=mock_api_key)
        assert provider.base_url == "https://api.openai.com/v1"

    def test_provider_name(self, mock_api_key):
        """Test provider name is correct."""
        provider = OpenAIProvider(api_key=mock_api_key)
        assert provider.provider_name == "openai"

    def test_endpoint(self, mock_api_key):
        """Test endpoint is correct."""
        provider = OpenAIProvider(api_key=mock_api_key)
        assert provider.get_endpoint() == "/chat/completions"

    def test_headers(self, mock_api_key):
        """Test headers include authorization."""
        provider = OpenAIProvider(api_key=mock_api_key)
        headers = provider.get_headers()

        assert headers["Authorization"] == f"Bearer {mock_api_key}"
        assert headers["Content-Type"] == "application/json"


class TestOpenAITransformRequest:
    """Tests for request transformation (should be passthrough)."""

    def test_request_passthrough(self, mock_api_key, openai_request):
        """Test request is passed through unchanged."""
        provider = OpenAIProvider(api_key=mock_api_key)

        result = provider.transform_request(openai_request)

        assert result == openai_request

    def test_preserves_all_fields(self, mock_api_key):
        """Test all request fields are preserved."""
        provider = OpenAIProvider(api_key=mock_api_key)
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "stop": ["\n"],
        }

        result = provider.transform_request(request)

        assert result == request


class TestOpenAITransformResponse:
    """Tests for response transformation (should be passthrough)."""

    def test_response_passthrough(self, mock_api_key, openai_response):
        """Test response is passed through unchanged."""
        provider = OpenAIProvider(api_key=mock_api_key)

        result = provider.transform_response(openai_response)

        assert result == openai_response


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""

    @responses.activate
    def test_full_chat_completion(self, mock_api_key, openai_request, openai_response):
        """Test full chat completion flow."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        provider = OpenAIProvider(api_key=mock_api_key)
        result = provider.chat_complete(openai_request)

        assert result["id"] == openai_response["id"]
        assert result["choices"][0]["message"]["content"] == "Hello! How can I assist you today?"
        assert result["provider"] == "openai"
        assert result["usage"]["total_tokens"] == 30

    @responses.activate
    def test_request_headers_sent(self, mock_api_key, openai_request, openai_response):
        """Test correct headers are sent in request."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json=openai_response,
            status=200,
        )

        provider = OpenAIProvider(api_key=mock_api_key)
        provider.chat_complete(openai_request)

        assert len(responses.calls) == 1
        request_headers = responses.calls[0].request.headers
        assert request_headers["Authorization"] == f"Bearer {mock_api_key}"
        assert request_headers["Content-Type"] == "application/json"

    @responses.activate
    def test_handles_api_error(self, mock_api_key, openai_request):
        """Test handling of API errors."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            status=401,
        )

        provider = OpenAIProvider(api_key=mock_api_key)
        result = provider.chat_complete(openai_request)

        assert "error" in result
        assert result["provider"] == "openai"
