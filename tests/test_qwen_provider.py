"""
Tests for Qwen (Alibaba DashScope) Provider — native API.
"""

import responses

from rezunate_llm_sdk.models import ChatCompletionRequest, ChatCompletionResponse, Provider
from rezunate_llm_sdk.providers.qwen_models import QwenRequest, QwenResponse
from rezunate_llm_sdk.providers.qwen_provider import QwenProvider


QWEN_FULL_URL = (
    "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)


class TestQwenProviderProperties:
    """Tests for Qwen provider properties."""

    def test_base_url(self, mock_api_key):
        provider = QwenProvider(api_key=mock_api_key)
        assert provider.base_url == "https://dashscope-intl.aliyuncs.com/api/v1"

    def test_provider_name(self, mock_api_key):
        provider = QwenProvider(api_key=mock_api_key)
        assert provider.provider_name == Provider.QWEN

    def test_endpoint(self, mock_api_key):
        provider = QwenProvider(api_key=mock_api_key)
        assert provider.get_endpoint() == "/services/aigc/text-generation/generation"

    def test_headers(self, mock_api_key):
        provider = QwenProvider(api_key=mock_api_key)
        headers = provider.get_headers()

        assert headers["Authorization"] == f"Bearer {mock_api_key}"
        assert headers["Content-Type"] == "application/json"


class TestQwenTransformRequest:
    """Tests for OpenAI -> DashScope native request transformation."""

    def test_basic_request_transformation(self, mock_api_key):
        """Messages move under input.messages; result_format defaults to 'message'."""
        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": "Hello!"}],
            }
        )

        result = provider.transform_request(req)

        assert isinstance(result, QwenRequest)
        assert result.model == "qwen-plus"
        assert len(result.input.messages) == 1
        assert result.input.messages[0].role == "user"
        assert result.input.messages[0].content == "Hello!"
        assert result.parameters.result_format == "message"

    def test_messages_nested_under_input(self, mock_api_key, sample_messages):
        """System + user messages all move into input.messages (no flattening)."""
        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {"model": "qwen-plus", "messages": sample_messages}
        )

        result = provider.transform_request(req)

        # DashScope keeps system in the messages array (unlike Anthropic/Google).
        assert len(result.input.messages) == 2
        assert result.input.messages[0].role == "system"
        assert result.input.messages[0].content == "You are a helpful assistant."
        assert result.input.messages[1].role == "user"

    def test_sampling_params_under_parameters(self, mock_api_key):
        """temperature/top_p/max_tokens move under parameters, not root."""
        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 200,
            }
        )

        result = provider.transform_request(req)

        assert result.parameters.temperature == 0.7
        assert result.parameters.top_p == 0.9
        assert result.parameters.max_tokens == 200

    def test_qwen_specific_extras_passed_through(self, mock_api_key):
        """top_k, seed, enable_search, repetition_penalty all reach parameters."""
        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": "Hi"}],
                "top_k": 50,
                "seed": 1234,
                "enable_search": True,
                "repetition_penalty": 1.05,
            }
        )

        result = provider.transform_request(req)

        assert result.parameters.top_k == 50
        assert result.parameters.seed == 1234
        assert result.parameters.enable_search is True
        assert result.parameters.repetition_penalty == 1.05

    def test_conversation_roles_preserved(self, mock_api_key, sample_conversation):
        """Multi-turn conversation preserves all roles inside input.messages."""
        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {"model": "qwen-plus", "messages": sample_conversation}
        )

        result = provider.transform_request(req)

        assert len(result.input.messages) == 4
        assert result.input.messages[0].role == "system"
        assert result.input.messages[1].role == "user"
        assert result.input.messages[2].role == "assistant"
        assert result.input.messages[3].role == "user"


class TestQwenTransformResponse:
    """Tests for DashScope native -> OpenAI response transformation."""

    def test_basic_response_transformation(self, mock_api_key, qwen_response):
        provider = QwenProvider(api_key=mock_api_key)

        resp_obj = QwenResponse.model_validate(qwen_response)
        result = provider.transform_response(resp_obj, model="qwen-plus")

        assert isinstance(result, ChatCompletionResponse)
        assert result.object == "chat.completion"
        assert result.model == "qwen-plus"
        assert len(result.choices) == 1
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].message.content == "Hello! How can I assist you today?"

    def test_finish_reason_stop(self, mock_api_key):
        provider = QwenProvider(api_key=mock_api_key)
        response = {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Done"},
                    }
                ]
            },
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        }

        resp_obj = QwenResponse.model_validate(response)
        result = provider.transform_response(resp_obj, model="qwen-plus")

        assert result.choices[0].finish_reason == "stop"

    def test_finish_reason_length(self, mock_api_key):
        provider = QwenProvider(api_key=mock_api_key)
        response = {
            "output": {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "Truncated..."},
                    }
                ]
            },
            "usage": {"input_tokens": 5, "output_tokens": 100, "total_tokens": 105},
        }

        resp_obj = QwenResponse.model_validate(response)
        result = provider.transform_response(resp_obj, model="qwen-plus")

        assert result.choices[0].finish_reason == "length"

    def test_finish_reason_tool_calls(self, mock_api_key):
        provider = QwenProvider(api_key=mock_api_key)
        response = {
            "output": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": "calling tool"},
                    }
                ]
            },
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }

        resp_obj = QwenResponse.model_validate(response)
        result = provider.transform_response(resp_obj, model="qwen-plus")

        assert result.choices[0].finish_reason == "tool_calls"

    def test_usage_unpacked_from_dashscope_fields(self, mock_api_key, qwen_response):
        """input_tokens/output_tokens map to prompt/completion tokens."""
        provider = QwenProvider(api_key=mock_api_key)

        resp_obj = QwenResponse.model_validate(qwen_response)
        result = provider.transform_response(resp_obj, model="qwen-plus")

        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    def test_request_id_used_as_response_id(self, mock_api_key, qwen_response):
        """DashScope request_id flows through to ChatCompletionResponse.id."""
        provider = QwenProvider(api_key=mock_api_key)

        resp_obj = QwenResponse.model_validate(qwen_response)
        result = provider.transform_response(resp_obj, model="qwen-plus")

        assert result.id == "req-test-123"

    def test_legacy_text_format_fallback(self, mock_api_key):
        """Older result_format='text' shape (output.text) still parses."""
        provider = QwenProvider(api_key=mock_api_key)
        response = {
            "output": {
                "text": "Hello!",
                "finish_reason": "stop",
            },
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

        resp_obj = QwenResponse.model_validate(response)
        result = provider.transform_response(resp_obj, model="qwen-plus")

        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].finish_reason == "stop"


class TestQwenIntegration:
    """Integration tests against the native DashScope endpoint (mocked HTTP)."""

    @responses.activate
    def test_full_chat_completion(self, mock_api_key, qwen_response):
        responses.add(responses.POST, QWEN_FULL_URL, json=qwen_response, status=200)

        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": "Hello!"}],
            }
        )

        result = provider.chat_complete(req)

        assert result.choices[0].message.content == "Hello! How can I assist you today?"
        assert result.provider == Provider.QWEN

    @responses.activate
    def test_request_body_uses_dashscope_shape(self, mock_api_key, qwen_response):
        """Outgoing JSON must have input.messages and parameters.result_format."""
        responses.add(responses.POST, QWEN_FULL_URL, json=qwen_response, status=200)

        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.5,
            }
        )

        provider.chat_complete(req)

        import json

        body = json.loads(responses.calls[0].request.body)
        assert body["model"] == "qwen-plus"
        assert "input" in body
        assert "messages" in body["input"]
        assert body["input"]["messages"][0]["role"] == "user"
        assert "parameters" in body
        assert body["parameters"]["result_format"] == "message"
        assert body["parameters"]["temperature"] == 0.5

    @responses.activate
    def test_bearer_auth_header_sent(self, mock_api_key, qwen_response):
        responses.add(responses.POST, QWEN_FULL_URL, json=qwen_response, status=200)

        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {"model": "qwen-plus", "messages": [{"role": "user", "content": "Hi"}]}
        )

        provider.chat_complete(req)

        headers = responses.calls[0].request.headers
        assert headers["Authorization"] == f"Bearer {mock_api_key}"

    @responses.activate
    def test_handles_api_error(self, mock_api_key):
        responses.add(
            responses.POST,
            QWEN_FULL_URL,
            json={
                "code": "AccessDenied.Unpurchased",
                "message": "Access to model denied.",
            },
            status=403,
        )

        provider = QwenProvider(api_key=mock_api_key)
        req = ChatCompletionRequest.model_validate(
            {"model": "qwen-plus", "messages": [{"role": "user", "content": "Hi"}]}
        )

        result = provider.chat_complete(req)

        assert result.error is not None
        assert result.error.code == 403
        assert result.provider == Provider.QWEN
