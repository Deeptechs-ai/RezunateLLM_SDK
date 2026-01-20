"""
Tests for Google (Gemini) Provider.
"""

import pytest
import responses
from providers.google_provider import GoogleProvider


class TestGoogleProviderProperties:
    """Tests for Google provider properties."""

    def test_base_url(self, mock_api_key):
        """Test base URL is correct."""
        provider = GoogleProvider(api_key=mock_api_key)
        assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"

    def test_provider_name(self, mock_api_key):
        """Test provider name is correct."""
        provider = GoogleProvider(api_key=mock_api_key)
        assert provider.provider_name == "google"

    def test_endpoint_includes_model(self, mock_api_key):
        """Test endpoint includes model name."""
        provider = GoogleProvider(api_key=mock_api_key)
        assert provider.get_endpoint("gemini-2.0-flash") == "/models/gemini-2.0-flash:generateContent"

    def test_headers(self, mock_api_key):
        """Test headers include API key."""
        provider = GoogleProvider(api_key=mock_api_key)
        headers = provider.get_headers()

        assert headers["x-goog-api-key"] == mock_api_key
        assert headers["Content-Type"] == "application/json"


class TestGoogleTransformRequest:
    """Tests for OpenAI -> Google request transformation."""

    def test_basic_request_transformation(self, mock_api_key):
        """Test basic request is transformed correctly."""
        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        result = provider.transform_request(openai_request)

        assert "contents" in result
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"] == [{"text": "Hello!"}]

    def test_system_message_extraction(self, mock_api_key, sample_messages):
        """Test system message is extracted to systemInstruction."""
        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": sample_messages,
        }

        result = provider.transform_request(openai_request)

        assert "systemInstruction" in result
        assert result["systemInstruction"]["parts"] == [{"text": "You are a helpful assistant."}]

        # Check user message is in contents
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"

    def test_role_mapping(self, mock_api_key, sample_conversation):
        """Test OpenAI roles are mapped to Google roles."""
        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": sample_conversation,
        }

        result = provider.transform_request(openai_request)

        # System extracted, 3 messages remain
        assert len(result["contents"]) == 3
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"  # assistant -> model
        assert result["contents"][2]["role"] == "user"

    def test_generation_config_temperature(self, mock_api_key):
        """Test temperature is added to generationConfig."""
        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
        }

        result = provider.transform_request(openai_request)

        assert result["generationConfig"]["temperature"] == 0.7

    def test_generation_config_max_tokens(self, mock_api_key):
        """Test max_tokens is mapped to maxOutputTokens."""
        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 100,
        }

        result = provider.transform_request(openai_request)

        assert result["generationConfig"]["maxOutputTokens"] == 100

    def test_top_k_passed_through(self, mock_api_key):
        """Test top_k is added to generationConfig."""
        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
            "top_k": 40,
        }

        result = provider.transform_request(openai_request)

        assert result["generationConfig"]["topK"] == 40

    def test_safety_settings_passed_through(self, mock_api_key):
        """Test safety_settings are passed through."""
        provider = GoogleProvider(api_key=mock_api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
            "safety_settings": safety_settings,
        }

        result = provider.transform_request(openai_request)

        assert result["safetySettings"] == safety_settings

    def test_tools_passed_through(self, mock_api_key):
        """Test tools are passed through."""
        provider = GoogleProvider(api_key=mock_api_key)
        tools = [{"function_declarations": [{"name": "get_weather"}]}]
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
            "tools": tools,
        }

        result = provider.transform_request(openai_request)

        assert result["tools"] == tools


class TestGoogleTransformResponse:
    """Tests for Google -> OpenAI response transformation."""

    def test_basic_response_transformation(self, mock_api_key, google_response):
        """Test basic response is transformed correctly."""
        provider = GoogleProvider(api_key=mock_api_key)

        result = provider.transform_response(google_response, model="gemini-2.0-flash")

        assert result["object"] == "chat.completion"
        assert result["model"] == "gemini-2.0-flash"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello! How can I assist you today?"

    def test_finish_reason_stop(self, mock_api_key):
        """Test STOP maps to stop."""
        provider = GoogleProvider(api_key=mock_api_key)
        response = {
            "candidates": [{
                "content": {"parts": [{"text": "Done"}], "role": "model"},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_max_tokens(self, mock_api_key):
        """Test MAX_TOKENS maps to length."""
        provider = GoogleProvider(api_key=mock_api_key)
        response = {
            "candidates": [{
                "content": {"parts": [{"text": "Truncated"}], "role": "model"},
                "finishReason": "MAX_TOKENS",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 100, "totalTokenCount": 110},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["finish_reason"] == "length"

    def test_finish_reason_safety(self, mock_api_key):
        """Test SAFETY maps to content_filter."""
        provider = GoogleProvider(api_key=mock_api_key)
        response = {
            "candidates": [{
                "content": {"parts": [{"text": ""}], "role": "model"},
                "finishReason": "SAFETY",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0, "totalTokenCount": 10},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["finish_reason"] == "content_filter"

    def test_usage_transformation(self, mock_api_key, google_response):
        """Test usage is transformed correctly."""
        provider = GoogleProvider(api_key=mock_api_key)

        result = provider.transform_response(google_response)

        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30

    def test_multiple_candidates(self, mock_api_key):
        """Test multiple candidates are transformed to choices."""
        provider = GoogleProvider(api_key=mock_api_key)
        response = {
            "candidates": [
                {"content": {"parts": [{"text": "Response 1"}], "role": "model"}, "finishReason": "STOP"},
                {"content": {"parts": [{"text": "Response 2"}], "role": "model"}, "finishReason": "STOP"},
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30},
        }

        result = provider.transform_response(response)

        assert len(result["choices"]) == 2
        assert result["choices"][0]["index"] == 0
        assert result["choices"][0]["message"]["content"] == "Response 1"
        assert result["choices"][1]["index"] == 1
        assert result["choices"][1]["message"]["content"] == "Response 2"

    def test_multiple_parts_concatenated(self, mock_api_key):
        """Test multiple text parts are concatenated."""
        provider = GoogleProvider(api_key=mock_api_key)
        response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "First "}, {"text": "Second"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20},
        }

        result = provider.transform_response(response)

        assert result["choices"][0]["message"]["content"] == "First Second"

    def test_response_has_created_timestamp(self, mock_api_key, google_response):
        """Test response includes created timestamp."""
        provider = GoogleProvider(api_key=mock_api_key)

        result = provider.transform_response(google_response)

        assert "created" in result
        assert isinstance(result["created"], int)

    def test_response_has_unique_id(self, mock_api_key, google_response):
        """Test response has a unique ID."""
        provider = GoogleProvider(api_key=mock_api_key)

        result = provider.transform_response(google_response)

        assert "id" in result
        assert result["id"].startswith("chatcmpl-")


class TestGoogleIntegration:
    """Integration tests for Google provider."""

    @responses.activate
    def test_full_chat_completion(self, mock_api_key, sample_messages, google_response):
        """Test full chat completion flow."""
        responses.add(
            responses.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=google_response,
            status=200,
        )

        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": sample_messages,
        }

        result = provider.chat_complete(openai_request)

        assert result["choices"][0]["message"]["content"] == "Hello! How can I assist you today?"
        assert result["provider"] == "google"

    @responses.activate
    def test_request_body_format(self, mock_api_key, sample_messages, google_response):
        """Test request body is in Google format."""
        responses.add(
            responses.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=google_response,
            status=200,
        )

        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": sample_messages,
        }

        provider.chat_complete(openai_request)

        # Verify request was transformed
        import json
        request_body = json.loads(responses.calls[0].request.body)
        assert "contents" in request_body
        assert "systemInstruction" in request_body

    @responses.activate
    def test_api_key_in_header(self, mock_api_key, google_response):
        """Test API key is sent in header, not URL."""
        responses.add(
            responses.POST,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            json=google_response,
            status=200,
        )

        provider = GoogleProvider(api_key=mock_api_key)
        openai_request = {
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        provider.chat_complete(openai_request)

        request_headers = responses.calls[0].request.headers
        assert request_headers["x-goog-api-key"] == mock_api_key
        # Ensure API key is NOT in URL
        assert "key=" not in responses.calls[0].request.url
