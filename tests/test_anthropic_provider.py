"""
Tests for Anthropic Provider.
"""

import responses

from providers.anthropic_provider import AnthropicProvider


class TestAnthropicProviderProperties:
    """Tests for Anthropic provider properties."""

    def test_base_url(self, mock_api_key):
        """Test base URL is correct."""
        provider = AnthropicProvider(api_key=mock_api_key)
        assert provider.base_url == "https://api.anthropic.com/v1"

    def test_provider_name(self, mock_api_key):
        """Test provider name is correct."""
        provider = AnthropicProvider(api_key=mock_api_key)
        assert provider.provider_name == "anthropic"

    def test_endpoint(self, mock_api_key):
        """Test endpoint is correct."""
        provider = AnthropicProvider(api_key=mock_api_key)
        assert provider.get_endpoint() == "/messages"

    def test_headers(self, mock_api_key):
        """Test headers include API key and version."""
        provider = AnthropicProvider(api_key=mock_api_key)
        headers = provider.get_headers()

        assert headers["x-api-key"] == mock_api_key
        assert headers["Content-Type"] == "application/json"
        assert "anthropic-version" in headers


class TestAnthropicTransformRequest:
    """Tests for OpenAI -> Anthropic request transformation."""

    def test_basic_request_transformation(self, mock_api_key):
        """Test basic request is transformed correctly."""
        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 100,
        }

        result = provider.transform_request(openai_request)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 100
        assert result["messages"] == [{"role": "user", "content": "Hello!"}]

    def test_system_message_extraction(self, mock_api_key, sample_messages):
        """Test system message is extracted to separate parameter."""
        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": sample_messages,
            "max_tokens": 100,
        }

        result = provider.transform_request(openai_request)

        assert result["system"] == "You are a helpful assistant."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello!"

    def test_default_max_tokens(self, mock_api_key):
        """Test default max_tokens is set when not provided."""
        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        result = provider.transform_request(openai_request)

        assert result["max_tokens"] == 1024

    def test_temperature_passed_through(self, mock_api_key):
        """Test temperature is passed through."""
        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
        }

        result = provider.transform_request(openai_request)

        assert result["temperature"] == 0.7

    def test_conversation_roles_mapped(self, mock_api_key, sample_conversation):
        """Test conversation roles are mapped correctly."""
        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": sample_conversation,
        }

        result = provider.transform_request(openai_request)

        # System message extracted
        assert result["system"] == "You are a helpful assistant."

        # Check remaining messages
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][2]["role"] == "user"

    def test_anthropic_specific_params(self, mock_api_key):
        """Test Anthropic-specific parameters are passed through."""
        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello!"}],
            "top_k": 40,
            "metadata": {"user_id": "123"},
        }

        result = provider.transform_request(openai_request)

        assert result["top_k"] == 40
        assert result["metadata"] == {"user_id": "123"}


class TestAnthropicTransformResponse:
    """Tests for Anthropic -> OpenAI response transformation."""

    def test_basic_response_transformation(self, mock_api_key, anthropic_response):
        """Test basic response is transformed correctly."""
        provider = AnthropicProvider(api_key=mock_api_key)

        result = provider.transform_response(anthropic_response)

        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello! How can I assist you today?"

    def test_stop_reason_mapping_end_turn(self, mock_api_key):
        """Test end_turn maps to stop."""
        provider = AnthropicProvider(api_key=mock_api_key)
        response = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["finish_reason"] == "stop"

    def test_stop_reason_mapping_max_tokens(self, mock_api_key):
        """Test max_tokens maps to length."""
        provider = AnthropicProvider(api_key=mock_api_key)
        response = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Truncated..."}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["finish_reason"] == "length"

    def test_stop_reason_mapping_tool_use(self, mock_api_key):
        """Test tool_use maps to tool_calls."""
        provider = AnthropicProvider(api_key=mock_api_key)
        response = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Using tool"}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_usage_transformation(self, mock_api_key, anthropic_response):
        """Test usage is transformed correctly."""
        provider = AnthropicProvider(api_key=mock_api_key)

        result = provider.transform_response(anthropic_response)

        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30

    def test_multiple_content_blocks(self, mock_api_key):
        """Test multiple text content blocks are concatenated."""
        provider = AnthropicProvider(api_key=mock_api_key)
        response = {
            "id": "msg_123",
            "content": [
                {"type": "text", "text": "First part. "},
                {"type": "text", "text": "Second part."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["message"]["content"] == "First part. Second part."

    def test_response_has_created_timestamp(self, mock_api_key, anthropic_response):
        """Test response includes created timestamp."""
        provider = AnthropicProvider(api_key=mock_api_key)

        result = provider.transform_response(anthropic_response)

        assert "created" in result
        assert isinstance(result["created"], int)


class TestAnthropicIntegration:
    """Integration tests for Anthropic provider."""

    @responses.activate
    def test_full_chat_completion(self, mock_api_key, sample_messages, anthropic_response):
        """Test full chat completion flow."""
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json=anthropic_response,
            status=200,
        )

        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": sample_messages,
            "max_tokens": 100,
        }

        result = provider.chat_complete(openai_request)

        assert result["choices"][0]["message"]["content"] == "Hello! How can I assist you today?"
        assert result["provider"] == "anthropic"

    @responses.activate
    def test_request_body_format(self, mock_api_key, sample_messages, anthropic_response):
        """Test request body is in Anthropic format."""
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json=anthropic_response,
            status=200,
        )

        provider = AnthropicProvider(api_key=mock_api_key)
        openai_request = {
            "model": "claude-sonnet-4-20250514",
            "messages": sample_messages,
            "max_tokens": 100,
        }

        provider.chat_complete(openai_request)

        # Verify request was transformed
        import json

        request_body = json.loads(responses.calls[0].request.body)
        assert "system" in request_body
        assert request_body["system"] == "You are a helpful assistant."
