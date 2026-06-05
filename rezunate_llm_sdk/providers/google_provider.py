"""
Google (Gemini) Provider.
Transforms OpenAI format <-> Google Gemini format.
"""

import json
import time
import uuid
from typing import Any

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
from rezunate_llm_sdk.providers.base import BaseProvider
from rezunate_llm_sdk.providers.endpoints import (
    GOOGLE_BASE_URL,
    GOOGLE_GENERATE_CONTENT_ENDPOINT,
)
from rezunate_llm_sdk.providers.google_models import (
    GoogleContentBlock,
    GoogleFunctionCall,
    GoogleFunctionCallingConfig,
    GoogleFunctionDeclaration,
    GoogleFunctionResponse,
    GoogleGenerationConfig,
    GoogleMessage,
    GoogleRequest,
    GoogleResponse,
    GoogleSystemInstruction,
    GoogleTool,
    GoogleToolConfig,
)


class GoogleProvider(BaseProvider):
    """
    Google Gemini Provider implementation.
    Handles transformation between OpenAI and Google Gemini formats.
    """

    response_model = GoogleResponse

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)

    @property
    def base_url(self) -> str:
        return GOOGLE_BASE_URL

    @property
    def provider_name(self) -> Provider:
        return Provider.GOOGLE

    def get_headers(self) -> dict[str, str]:
        return {
            constants.CONTENT_TYPE_HEADER: constants.APPLICATION_JSON,
            constants.GOOGLE_API_KEY_HEADER: self.api_key,
        }

    def get_endpoint(self, model: str = None) -> str:
        return GOOGLE_GENERATE_CONTENT_ENDPOINT.format(model=model)

    def transform_request(self, request: ChatCompletionRequest) -> GoogleRequest:
        """Transform OpenAI format request to Google Gemini format.

        Converts messages to Google's contents format, maps ``assistant`` role to
        ``model``, extracts system messages to ``systemInstruction``, translates
        tool calls and tool messages to ``functionCall`` / ``functionResponse``
        parts, and maps generation parameters.
        """
        system_content: str | None = None
        google_messages: list[GoogleMessage] = []

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_content = msg.content
                continue

            if msg.role == Role.TOOL:
                try:
                    response_payload: Any = json.loads(msg.content) if msg.content else {}
                except (TypeError, json.JSONDecodeError):
                    response_payload = {"result": msg.content}
                if not isinstance(response_payload, dict):
                    response_payload = {"result": response_payload}
                google_messages.append(
                    GoogleMessage(
                        role="user",
                        parts=[
                            GoogleContentBlock(
                                functionResponse=GoogleFunctionResponse(
                                    name=msg.name or msg.tool_call_id or "tool",
                                    response=response_payload,
                                )
                            )
                        ],
                    )
                )
                continue

            google_role = "model" if msg.role == Role.ASSISTANT else "user"
            parts = _message_to_google_parts(msg)
            google_messages.append(GoogleMessage(role=google_role, parts=parts))

        system_instruction = (
            GoogleSystemInstruction(parts=[GoogleContentBlock(text=system_content)])
            if system_content
            else None
        )

        generation_config = None
        if (
            request.temperature is not None
            or request.max_tokens is not None
            or getattr(request, "top_k", None) is not None
        ):
            generation_config = GoogleGenerationConfig(
                temperature=request.temperature,
                maxOutputTokens=request.max_tokens,
                topK=getattr(request, "top_k", None),
            )

        return GoogleRequest(
            contents=google_messages,
            systemInstruction=system_instruction,
            generationConfig=generation_config,
            safetySettings=getattr(request, "safety_settings", None),
            tools=_translate_tools(request.tools),
            toolConfig=_translate_tool_choice(request.tool_choice),
        )

    def transform_response(
        self, response: GoogleResponse, model: str | None = None
    ) -> ChatCompletionResponse:
        """Transform Google Gemini format response to OpenAI format.

        Walks ``candidates[].content.parts`` to collect text into
        ``message.content`` and ``functionCall`` parts into ``message.tool_calls``.
        ``finish_reason`` is forced to ``TOOL_CALLS`` whenever any function call
        is present (Gemini still returns ``STOP`` in that case).
        """
        google_response = response

        choices: list[Choice] = []
        for idx, candidate in enumerate(google_response.candidates):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            if candidate.content:
                for part in candidate.content.parts:
                    if part.text:
                        text_parts.append(part.text)
                    if part.functionCall:
                        fc = part.functionCall
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                function=FunctionCall(
                                    name=fc.name,
                                    arguments=json.dumps(fc.args or {}),
                                ),
                            )
                        )

            finish_reason = FINISH_REASON_MAP.get(candidate.finishReason, FinishReason.STOP)
            if tool_calls:
                finish_reason = FinishReason.TOOL_CALLS

            content = "".join(text_parts) if text_parts else None

            choices.append(
                Choice(
                    index=idx,
                    message=ResponseMessage(
                        content=content,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            )

        usage_meta = google_response.usageMetadata
        prompt_tokens = usage_meta.promptTokenCount
        completion_tokens = usage_meta.candidatesTokenCount
        total_tokens = usage_meta.totalTokenCount or (prompt_tokens + completion_tokens)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            provider=self.provider_name,
        )


def _message_to_google_parts(msg: Message) -> list[GoogleContentBlock]:
    """Render a normalized assistant/user Message as a list of Gemini parts."""
    parts: list[GoogleContentBlock] = []
    if msg.content:
        parts.append(GoogleContentBlock(text=msg.content))
    for call in msg.tool_calls or []:
        try:
            args = json.loads(call.function.arguments) if call.function.arguments else {}
        except json.JSONDecodeError:
            args = {"_raw": call.function.arguments}
        parts.append(
            GoogleContentBlock(functionCall=GoogleFunctionCall(name=call.function.name, args=args))
        )
    if not parts:
        parts.append(GoogleContentBlock(text=""))
    return parts


def _translate_tools(tools: list[Tool] | None) -> list[GoogleTool] | None:
    """Translate OpenAI-shaped tools into Gemini's ``functionDeclarations`` shape."""
    if not tools:
        return None
    declarations = [
        GoogleFunctionDeclaration(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=tool.function.parameters or {"type": "object", "properties": {}},
        )
        for tool in tools
    ]
    return [GoogleTool(functionDeclarations=declarations)]


def _translate_tool_choice(
    tool_choice: str | ToolChoiceOption | None,
) -> GoogleToolConfig | None:
    """Translate OpenAI-shaped ``tool_choice`` into Gemini's ``toolConfig``."""
    if tool_choice is None:
        return None
    mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
    if isinstance(tool_choice, str):
        return GoogleToolConfig(
            functionCallingConfig=GoogleFunctionCallingConfig(mode=mode_map[tool_choice])
        )
    if isinstance(tool_choice, ToolChoiceOption):
        return GoogleToolConfig(
            functionCallingConfig=GoogleFunctionCallingConfig(
                mode="ANY", allowedFunctionNames=[tool_choice.function.name]
            )
        )
    return None
