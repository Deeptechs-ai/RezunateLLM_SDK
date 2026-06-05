"""
Anthropic Provider.
Transforms OpenAI format to Anthropic format
"""

import json
import time
import uuid

import rezunate_llm_sdk.constants as constants
from rezunate_llm_sdk.models import (
    FINISH_REASON_MAP,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    FinishReason,
    FunctionCall,
    Message,
    Provider,
    ResponseMessage,
    Role,
    Tool,
    ToolCall,
    ToolChoiceOption,
    Usage,
)
from rezunate_llm_sdk.providers.anthropic_models import (
    AnthropicAnyToolChoice,
    AnthropicAutoToolChoice,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicSpecificToolChoice,
    AnthropicTextBlock,
    AnthropicTool,
    AnthropicToolChoice,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
)
from rezunate_llm_sdk.providers.base import BaseProvider
from rezunate_llm_sdk.providers.endpoints import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_DEFAULT_VERSION,
    ANTHROPIC_MESSAGES_ENDPOINT,
)


class AnthropicProvider(BaseProvider):
    """
    Anthropic Provider implementation.
    Handles transformation between OpenAI and Anthropic formats.
    """

    response_model = AnthropicResponse

    @property
    def base_url(self) -> str:
        return ANTHROPIC_BASE_URL

    @property
    def provider_name(self) -> Provider:
        return Provider.ANTHROPIC

    def get_headers(self) -> dict[str, str]:
        return {
            constants.API_KEY_HEADER: self.api_key,
            constants.CONTENT_TYPE_HEADER: constants.APPLICATION_JSON,
            constants.ANTHROPIC_VERSION_HEADER: ANTHROPIC_DEFAULT_VERSION,
        }

    def get_endpoint(self, model: str | None = None) -> str:
        return ANTHROPIC_MESSAGES_ENDPOINT

    def transform_request(self, request: ChatCompletionRequest) -> AnthropicRequest:
        """
        Transform OpenAI format request to Anthropic format.

        Key differences:
        - OpenAI: system message in messages array
        - Anthropic: system message is separate parameter
        - OpenAI: max_tokens optional
        - Anthropic: max_tokens required
        - OpenAI tool calls live on ``message.tool_calls`` with a ``tool`` role for
          results; Anthropic encodes them as ``tool_use`` / ``tool_result`` content
          blocks under ``assistant`` and ``user`` roles respectively.
        """
        system_content: str | None = None
        anthropic_messages: list[AnthropicMessage] = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_content = msg.content
                continue

            if msg.role == Role.TOOL:
                anthropic_messages.append(
                    AnthropicMessage(
                        role="user",
                        content=[
                            AnthropicToolResultBlock(
                                tool_use_id=msg.tool_call_id or "",
                                content=msg.content or "",
                            )
                        ],
                    )
                )
                continue

            anthropic_role = "assistant" if msg.role == Role.ASSISTANT else "user"
            blocks = _message_to_anthropic_blocks(msg)
            if blocks is None:
                anthropic_messages.append(
                    AnthropicMessage(role=anthropic_role, content=msg.content or "")
                )
            else:
                anthropic_messages.append(AnthropicMessage(role=anthropic_role, content=blocks))

        return AnthropicRequest(
            model=request.model,
            max_tokens=request.max_tokens or 1024,
            messages=anthropic_messages,
            system=system_content,
            temperature=request.temperature,
            top_k=getattr(request, "top_k", None),
            metadata=getattr(request, "metadata", None),
            tools=_translate_tools(request.tools),
            tool_choice=_translate_tool_choice(request.tool_choice),
        )

    def transform_response(
        self, response: AnthropicResponse, model: str | None = None
    ) -> ChatCompletionResponse:
        """
        Transform Anthropic format response to OpenAI format.

        Concatenates text blocks into ``message.content`` and collects ``tool_use``
        blocks into ``message.tool_calls``.
        """
        anthropic_response = response

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in anthropic_response.content:
            if isinstance(block, AnthropicTextBlock):
                text_parts.append(block.text)
            elif isinstance(block, AnthropicToolUseBlock):
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        function=FunctionCall(
                            name=block.name,
                            arguments=json.dumps(block.input or {}),
                        ),
                    )
                )

        finish_reason = FINISH_REASON_MAP.get(anthropic_response.stop_reason, FinishReason.STOP)
        content = "".join(text_parts) if text_parts else None

        input_tokens = anthropic_response.usage.input_tokens
        output_tokens = anthropic_response.usage.output_tokens

        return ChatCompletionResponse(
            id=anthropic_response.id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=anthropic_response.model or model,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(
                        content=content,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            provider=self.provider_name,
        )


def _message_to_anthropic_blocks(
    msg: Message,
) -> list[AnthropicTextBlock | AnthropicToolUseBlock] | None:
    """Build a list of Anthropic content blocks from a normalized assistant Message.

    Returns ``None`` when the message has no tool calls (caller should fall back
    to the simpler string-content shape).
    """
    if not msg.tool_calls:
        return None

    blocks: list[AnthropicTextBlock | AnthropicToolUseBlock] = []
    if msg.content:
        blocks.append(AnthropicTextBlock(text=msg.content))
    for call in msg.tool_calls:
        try:
            arguments = json.loads(call.function.arguments) if call.function.arguments else {}
        except json.JSONDecodeError:
            arguments = {"_raw": call.function.arguments}
        blocks.append(AnthropicToolUseBlock(id=call.id, name=call.function.name, input=arguments))
    return blocks


def _translate_tools(tools: list[Tool] | None) -> list[AnthropicTool] | None:
    """Translate OpenAI-shaped tools into Anthropic's tool schema."""
    if not tools:
        return None
    return [
        AnthropicTool(
            name=tool.function.name,
            description=tool.function.description or "",
            input_schema=tool.function.parameters or {"type": "object", "properties": {}},
        )
        for tool in tools
    ]


def _translate_tool_choice(
    tool_choice: str | ToolChoiceOption | None,
) -> AnthropicToolChoice | None:
    """Translate OpenAI-shaped ``tool_choice`` into Anthropic's format."""
    if tool_choice is None or tool_choice == "none":
        return None
    if tool_choice == "auto":
        return AnthropicAutoToolChoice()
    if tool_choice == "required":
        return AnthropicAnyToolChoice()
    if isinstance(tool_choice, ToolChoiceOption):
        return AnthropicSpecificToolChoice(name=tool_choice.function.name)
    return None
