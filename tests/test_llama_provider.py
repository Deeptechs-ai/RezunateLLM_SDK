"""
Tests for Llama (Meta) Provider — native API.
"""

import responses

from rezunate_llm_sdk.models import ChatCompletionRequest, ChatCompletionResponse, Provider
from rezunate_llm_sdk.providers.llama_models import LlamaRequest, LlamaResponse
from rezunate_llm_sdk.providers.llama_provider import LlamaProvider


LLAMA_FULL_URL = "https://api.llama.com/v1/chat/completions"
LLAMA_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"


class TestLlamaProviderProperties:
    """Tests for Llama provider properties."""

    def test_base_url(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        assert provider.base_url == "https://api.llama.com/v1"

    def test_provider_name(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        assert provider.provider_name == Provider.LLAMA

    def test_endpoint(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        assert provider.get_endpoint() == "/chat/completions"

    def test_headers(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        headers = provider.get_headers()

        assert headers["Authorization"] == f"Bearer {mock_api_key}"
        assert headers["Content-Type"] == "application/json"


class TestLlamaTransformRequest:
    """Tests for OpenAI -> Meta Llama native request transformation."""

    def test_basic_request_transformation(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": LLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hello!"}],
            }
        )

        result = provider.transform_request(req)

        assert isinstance(result, LlamaRequest)
        assert result.model == LLAMA_MODEL
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello!"

    def test_max_tokens_mapped_to_max_completion_tokens(self, mock_api_key):
        """OpenAI ``max_tokens`` maps to Meta's ``max_completion_tokens``."""
        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": LLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 150,
            }
        )

        result = provider.transform_request(req)

        assert result.max_completion_tokens == 150

    def test_temperature_passed_through(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": LLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.6,
            }
        )

        result = provider.transform_request(req)

        assert result.temperature == 0.6

    def test_meta_specific_extras(self, mock_api_key):
        """top_k and repetition_penalty are Meta-specific fields."""
        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": LLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "top_k": 40,
                "repetition_penalty": 1.1,
            }
        )

        result = provider.transform_request(req)

        assert result.top_k == 40
        assert result.repetition_penalty == 1.1

    def test_conversation_roles_preserved(self, mock_api_key, sample_conversation):
        """System / user / assistant roles all preserved in messages array."""
        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {"model": LLAMA_MODEL, "messages": sample_conversation}
        )

        result = provider.transform_request(req)

        assert len(result.messages) == 4
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "assistant"
        assert result.messages[3].role == "user"

    def test_tools_and_response_format_passed_through(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        response_format = {"type": "json_object"}
        req = ChatCompletionRequest.model_validate(
            {
                "model": LLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": tools,
                "response_format": response_format,
            }
        )

        result = provider.transform_request(req)

        assert result.tools == tools
        assert result.response_format == response_format


class TestLlamaTransformResponse:
    """Tests for Meta Llama native -> OpenAI response transformation."""

    def test_basic_response_transformation(self, mock_api_key, llama_response):
        """completion_message.content.text becomes choices[0].message.content."""
        provider = LlamaProvider(api_key=mock_api_key)

        resp_obj = LlamaResponse.model_validate(llama_response)
        result = provider.transform_response(resp_obj, model=LLAMA_MODEL)

        assert isinstance(result, ChatCompletionResponse)
        assert result.object == "chat.completion"
        assert result.model == LLAMA_MODEL
        assert len(result.choices) == 1
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "Hello! How can I assist you today?"

    def test_finish_reason_stop(self, mock_api_key, llama_response):
        """stop_reason from completion_message becomes choices[0].finish_reason."""
        provider = LlamaProvider(api_key=mock_api_key)

        resp_obj = LlamaResponse.model_validate(llama_response)
        result = provider.transform_response(resp_obj, model=LLAMA_MODEL)

        assert result.choices[0].finish_reason == "stop"

    def test_finish_reason_length(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        response = {
            "completion_message": {
                "role": "assistant",
                "content": {"type": "text", "text": "Truncated..."},
                "stop_reason": "length",
            },
            "metrics": [
                {"metric": "num_prompt_tokens", "value": 10, "unit": "tokens"},
                {"metric": "num_completion_tokens", "value": 100, "unit": "tokens"},
            ],
        }

        resp_obj = LlamaResponse.model_validate(response)
        result = provider.transform_response(resp_obj, model=LLAMA_MODEL)

        assert result.choices[0].finish_reason == "length"

    def test_finish_reason_tool_calls(self, mock_api_key):
        provider = LlamaProvider(api_key=mock_api_key)
        response = {
            "completion_message": {
                "role": "assistant",
                "content": {"type": "text", "text": "calling tool"},
                "stop_reason": "tool_calls",
            },
            "metrics": [],
        }

        resp_obj = LlamaResponse.model_validate(response)
        result = provider.transform_response(resp_obj, model=LLAMA_MODEL)

        assert result.choices[0].finish_reason == "tool_calls"

    def test_metrics_array_unpacked_into_usage(self, mock_api_key, llama_response):
        """metrics array entries map to prompt/completion/total tokens."""
        provider = LlamaProvider(api_key=mock_api_key)

        resp_obj = LlamaResponse.model_validate(llama_response)
        result = provider.transform_response(resp_obj, model=LLAMA_MODEL)

        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    def test_total_tokens_derived_when_missing(self, mock_api_key):
        """If num_total_tokens metric is absent, fall back to prompt + completion."""
        provider = LlamaProvider(api_key=mock_api_key)
        response = {
            "completion_message": {
                "role": "assistant",
                "content": {"type": "text", "text": "Hi"},
                "stop_reason": "stop",
            },
            "metrics": [
                {"metric": "num_prompt_tokens", "value": 8, "unit": "tokens"},
                {"metric": "num_completion_tokens", "value": 5, "unit": "tokens"},
            ],
        }

        resp_obj = LlamaResponse.model_validate(response)
        result = provider.transform_response(resp_obj, model=LLAMA_MODEL)

        assert result.usage.total_tokens == 13

    def test_id_flows_through(self, mock_api_key, llama_response):
        provider = LlamaProvider(api_key=mock_api_key)

        resp_obj = LlamaResponse.model_validate(llama_response)
        result = provider.transform_response(resp_obj, model=LLAMA_MODEL)

        assert result.id == "msg-test-123"


class TestLlamaIntegration:
    """Integration tests against Meta's native endpoint (mocked HTTP)."""

    @responses.activate
    def test_full_chat_completion(self, mock_api_key, llama_response):
        responses.add(responses.POST, LLAMA_FULL_URL, json=llama_response, status=200)

        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": LLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hello!"}],
            }
        )

        result = provider.chat_complete(req)

        assert result.choices[0].message.content == "Hello! How can I assist you today?"
        assert result.provider == Provider.LLAMA
        assert result.usage.total_tokens == 30

    @responses.activate
    def test_request_body_uses_max_completion_tokens(self, mock_api_key, llama_response):
        """Outgoing JSON uses Meta's ``max_completion_tokens`` field name."""
        responses.add(responses.POST, LLAMA_FULL_URL, json=llama_response, status=200)

        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": LLAMA_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 75,
            }
        )

        provider.chat_complete(req)

        import json
        body = json.loads(responses.calls[0].request.body)
        assert body["model"] == LLAMA_MODEL
        assert body["max_completion_tokens"] == 75
        # Meta uses max_completion_tokens, not OpenAI's max_tokens
        assert "max_tokens" not in body

    @responses.activate
    def test_bearer_auth_header_sent(self, mock_api_key, llama_response):
        responses.add(responses.POST, LLAMA_FULL_URL, json=llama_response, status=200)

        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {"model": LLAMA_MODEL, "messages": [{"role": "user", "content": "Hi"}]}
        )

        provider.chat_complete(req)

        headers = responses.calls[0].request.headers
        assert headers["Authorization"] == f"Bearer {mock_api_key}"

    @responses.activate
    def test_handles_api_error(self, mock_api_key):
        responses.add(
            responses.POST,
            LLAMA_FULL_URL,
            json={"message": "Invalid API key"},
            status=401,
        )

        provider = LlamaProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {"model": LLAMA_MODEL, "messages": [{"role": "user", "content": "Hi"}]}
        )

        result = provider.chat_complete(req)

        assert result.error is not None
        assert result.error.code == 401
        assert result.provider == Provider.LLAMA
