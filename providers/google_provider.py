"""
Google (Gemini) Provider.
Transforms OpenAI format <-> Google Gemini format.
"""

from typing import Dict, Any
import time
import uuid
from providers.base import BaseProvider


class GoogleProvider(BaseProvider):
    """
    Google Gemini Provider implementation.
    Handles transformation between OpenAI and Google Gemini formats.
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)

    @property
    def base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    @property
    def provider_name(self) -> str:
        return "google"

    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

    def get_endpoint(self, model: str = None) -> str:
        return f"/models/{model}:generateContent"

    def transform_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform OpenAI format request to Google Gemini format.

        OpenAI format:
        {
            "model": "...",
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ]
        }

        Google Gemini format:
        {
            "contents": [
                {"role": "user", "parts": [{"text": "..."}]}
            ],
            "systemInstruction": {"parts": [{"text": "..."}]},
            "generationConfig": {...}
        }
        """
        google_request = {
            "contents": [],
            "generationConfig": {}
        }

        # Extract system message and regular messages
        messages = request.get("messages", [])
        system_content = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            else:
                # Map OpenAI roles to Google roles
                # Google uses "user" and "model" (not "assistant")
                google_role = "model" if role == "assistant" else "user"
                google_request["contents"].append({
                    "role": google_role,
                    "parts": [{"text": content}]
                })

        # Add system instruction if present
        if system_content:
            google_request["systemInstruction"] = {
                "parts": [{"text": system_content}]
            }

        # Map generation config
        generation_config = {}

        if "temperature" in request:
            generation_config["temperature"] = request["temperature"]

        if "max_tokens" in request:
            generation_config["maxOutputTokens"] = request["max_tokens"]

        # Pass through Google-specific generation config params
        # top_k is supported by Google but not OpenAI
        if "top_k" in request:
            generation_config["topK"] = request["top_k"]

        if generation_config:
            google_request["generationConfig"] = generation_config

        # Pass through Google-specific top-level params
        # These go directly in the request, not in generationConfig
        if "safety_settings" in request:
            google_request["safetySettings"] = request["safety_settings"]

        if "tools" in request:
            google_request["tools"] = request["tools"]

        return google_request

    def transform_response(self, response: Dict[str, Any], model: str = None) -> Dict[str, Any]:
        """
        Transform Google Gemini format response to OpenAI format.

        Google Gemini response:
        {
            "candidates": [{
                "content": {
                    "parts": [{"text": "..."}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": X,
                "candidatesTokenCount": Y,
                "totalTokenCount": Z
            }
        }

        OpenAI format:
        {
            "id": "chatcmpl-...",
            "object": "chat.completion",
            "created": timestamp,
            "model": "...",
            "choices": [{...}],
            "usage": {...}
        }
        """
        # Extract content from candidates
        choices = []
        candidates = response.get("candidates", [])

        for idx, candidate in enumerate(candidates):
            content = ""
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    content += part["text"]

            # Map Google finish reason to OpenAI
            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "stop"
            }
            finish_reason = finish_reason_map.get(
                candidate.get("finishReason", "STOP"),
                "stop"
            )

            choices.append({
                "index": idx,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": finish_reason
            })

        # Extract usage metadata
        usage_metadata = response.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + completion_tokens)

        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        return openai_response
