"""Tool-call round-trip tests across providers.

Covers:
- Role.TOOL exists on the enum
- OpenAI-shaped tool messages serialize correctly
- Anthropic provider: OpenAI tools/tool_choice -> Anthropic tools/tool_choice,
  assistant tool_calls -> tool_use blocks, Role.TOOL -> tool_result block,
  response tool_use blocks -> normalized ToolCall list
- Google provider: OpenAI tools -> functionDeclarations, tool_choice -> toolConfig,
  assistant tool_calls -> functionCall parts, Role.TOOL -> functionResponse part,
  response functionCall parts -> normalized ToolCall list
"""

import json

from rezunate_llm_sdk.models import (
    ChatCompletionRequest,
    FunctionCall,
    Message,
    Role,
    ToolCall,
)
from rezunate_llm_sdk.providers.anthropic_models import (
    AnthropicResponse,
    AnthropicTextBlock,
    AnthropicToolUseBlock,
)
from rezunate_llm_sdk.providers.anthropic_provider import AnthropicProvider
from rezunate_llm_sdk.providers.google_models import (
    GoogleCandidate,
    GoogleContentBlock,
    GoogleMessage,
    GoogleResponse,
    GoogleUsage,
)
from rezunate_llm_sdk.providers.google_provider import GoogleProvider

API_KEY = "test-key"


# --- enum + model shape ----------------------------------------------------


def test_role_tool_exists():
    assert Role.TOOL == "tool"
    assert Role("tool") is Role.TOOL


def test_message_serializes_tool_call():
    msg = Message(
        role=Role.ASSISTANT,
        tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="add", arguments='{"a":1}'))],
    )
    dumped = msg.model_dump(exclude_none=True)
    assert dumped["role"] == "assistant"
    assert dumped["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a":1}'}}
    ]
    assert "content" not in dumped


def test_message_serializes_tool_result():
    msg = Message(role=Role.TOOL, content="42", tool_call_id="call_1")
    dumped = msg.model_dump(exclude_none=True)
    assert dumped == {"role": "tool", "content": "42", "tool_call_id": "call_1"}


# --- Anthropic provider ----------------------------------------------------


def _anthropic() -> AnthropicProvider:
    return AnthropicProvider(api_key=API_KEY)


def test_anthropic_translates_openai_tools():
    request = ChatCompletionRequest(
        model="claude-sonnet-4-5",
        messages=[Message(role=Role.USER, content="hi")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice="required",
    )
    out = _anthropic().transform_request(request)
    assert out.tools is not None and len(out.tools) == 1
    tool = out.tools[0]
    assert tool.name == "get_weather"
    assert tool.description == "Look up the weather"
    assert tool.input_schema == {"type": "object", "properties": {"city": {"type": "string"}}}
    assert out.tool_choice is not None and out.tool_choice.type == "any"


def test_anthropic_tool_choice_specific_function():
    request = ChatCompletionRequest(
        model="claude-sonnet-4-5",
        messages=[Message(role=Role.USER, content="hi")],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    out = _anthropic().transform_request(request)
    assert out.tool_choice is not None
    assert out.tool_choice.type == "tool"
    assert out.tool_choice.name == "get_weather"


def test_anthropic_serializes_assistant_tool_calls_as_blocks():
    request = ChatCompletionRequest(
        model="claude-sonnet-4-5",
        messages=[
            Message(role=Role.USER, content="What's the weather in Tokyo?"),
            Message(
                role=Role.ASSISTANT,
                content="I'll check.",
                tool_calls=[
                    ToolCall(
                        id="toolu_1",
                        function=FunctionCall(name="get_weather", arguments='{"city":"Tokyo"}'),
                    )
                ],
            ),
            Message(role=Role.TOOL, content="sunny, 22C", tool_call_id="toolu_1"),
        ],
    )
    out = _anthropic().transform_request(request)
    # Three messages: user (string content), assistant (blocks), user (tool_result).
    assert len(out.messages) == 3
    assert out.messages[0].role == "user"
    assert out.messages[0].content == "What's the weather in Tokyo?"

    assistant_blocks = out.messages[1].content
    assert isinstance(assistant_blocks, list)
    assert isinstance(assistant_blocks[0], AnthropicTextBlock)
    assert assistant_blocks[0].text == "I'll check."
    assert isinstance(assistant_blocks[1], AnthropicToolUseBlock)
    assert assistant_blocks[1].id == "toolu_1"
    assert assistant_blocks[1].name == "get_weather"
    assert assistant_blocks[1].input == {"city": "Tokyo"}

    tool_msg_blocks = out.messages[2].content
    assert out.messages[2].role == "user"
    assert isinstance(tool_msg_blocks, list)
    assert tool_msg_blocks[0].type == "tool_result"
    assert tool_msg_blocks[0].tool_use_id == "toolu_1"
    assert tool_msg_blocks[0].content == "sunny, 22C"


def test_anthropic_parses_tool_use_response():
    anthropic_resp = AnthropicResponse.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_xyz",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 12},
        }
    )
    result = _anthropic().transform_response(anthropic_resp, model="claude-sonnet-4-5")

    assert result.choices[0].finish_reason == "tool_calls"
    msg = result.choices[0].message
    assert msg.content == "Let me check."
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "toolu_xyz"
    assert tc.type == "function"
    assert tc.function.name == "get_weather"
    assert json.loads(tc.function.arguments) == {"city": "Tokyo"}


# --- Google provider --------------------------------------------------------


def _google() -> GoogleProvider:
    return GoogleProvider(api_key=API_KEY)


def test_google_translates_openai_tools_into_function_declarations():
    request = ChatCompletionRequest(
        model="gemini-2.0-flash",
        messages=[Message(role=Role.USER, content="hi")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice="required",
    )
    out = _google().transform_request(request)
    assert out.tools is not None and len(out.tools) == 1
    decl = out.tools[0].functionDeclarations[0]
    assert decl.name == "get_weather"
    assert decl.description == "Look up the weather"
    assert decl.parameters == {"type": "object", "properties": {"city": {"type": "string"}}}
    assert out.toolConfig is not None
    assert out.toolConfig.functionCallingConfig.mode == "ANY"
    assert out.toolConfig.functionCallingConfig.allowedFunctionNames is None


def test_google_tool_choice_specific_function():
    request = ChatCompletionRequest(
        model="gemini-2.0-flash",
        messages=[Message(role=Role.USER, content="hi")],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    out = _google().transform_request(request)
    assert out.toolConfig is not None
    assert out.toolConfig.functionCallingConfig.mode == "ANY"
    assert out.toolConfig.functionCallingConfig.allowedFunctionNames == ["get_weather"]


def test_google_serializes_assistant_tool_calls_and_tool_results():
    request = ChatCompletionRequest(
        model="gemini-2.0-flash",
        messages=[
            Message(role=Role.USER, content="What's the weather in Tokyo?"),
            Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="call_xyz",
                        function=FunctionCall(name="get_weather", arguments='{"city":"Tokyo"}'),
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content='{"temp":"22C"}',
                tool_call_id="call_xyz",
                name="get_weather",
            ),
        ],
    )
    out = _google().transform_request(request)
    assert len(out.contents) == 3
    assert out.contents[0].role == "user"
    assert out.contents[0].parts[0].text == "What's the weather in Tokyo?"

    assistant_parts = out.contents[1].parts
    assert out.contents[1].role == "model"
    fc_part = assistant_parts[0].functionCall
    assert fc_part is not None
    assert fc_part.name == "get_weather"
    assert fc_part.args == {"city": "Tokyo"}

    tool_parts = out.contents[2].parts
    assert out.contents[2].role == "user"
    fr_part = tool_parts[0].functionResponse
    assert fr_part is not None
    assert fr_part.name == "get_weather"
    assert fr_part.response == {"temp": "22C"}


def test_google_parses_function_call_response():
    google_resp = GoogleResponse(
        candidates=[
            GoogleCandidate(
                content=GoogleMessage(
                    role="model",
                    parts=[
                        GoogleContentBlock(text="Looking it up."),
                        GoogleContentBlock(
                            functionCall={"name": "get_weather", "args": {"city": "Tokyo"}}
                        ),
                    ],
                ),
                finishReason="STOP",
            )
        ],
        usageMetadata=GoogleUsage(promptTokenCount=10, candidatesTokenCount=5, totalTokenCount=15),
    )
    result = _google().transform_response(google_resp, model="gemini-2.0-flash")
    msg = result.choices[0].message
    assert msg.content == "Looking it up."
    assert msg.tool_calls is not None and len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.type == "function"
    assert tc.function.name == "get_weather"
    assert json.loads(tc.function.arguments) == {"city": "Tokyo"}
    # finish_reason is forced to TOOL_CALLS when functionCall parts are present.
    assert result.choices[0].finish_reason == "tool_calls"
