"""Gateway - Routes chat completion requests to providers."""

import logging
import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from rezunate_llm_sdk.api import get_prompt as _api_get_prompt
from rezunate_llm_sdk.api import scan_text as _api_scan_text
from rezunate_llm_sdk.client import RouterClient
from rezunate_llm_sdk.guardrails import (
    ServerGuardrailsError,
    check_guardrails,
    load_guardrails,
)
from rezunate_llm_sdk.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    GuardrailDirection,
    GuardrailsConfig,
    Provider,
    ScanResponse,
    ServerGuardrailsConfig,
)
from rezunate_llm_sdk.prompts import render_prompt
from rezunate_llm_sdk.providers import get_provider, list_providers

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# Guardrails file path
GUARDRAILS_FILE = os.getenv("GUARDRAILS_FILE_PATH")


@lru_cache(maxsize=1)
def _get_automatic_config() -> GuardrailsConfig | None:
    """Load default guardrails from guardrails.example.yaml if it exists."""
    if not GUARDRAILS_FILE:
        return None
    path = Path(GUARDRAILS_FILE)
    if not path.exists():
        return None
    try:
        config = load_guardrails(path)
        logger.info("Automatically loaded guardrails from %s", path)
        return config
    except Exception as e:
        logger.warning("Failed to load automatic guardrails from %s: %s", path, e)
    return None


def chat_complete(
    provider: str | Provider,
    api_key: str,
    request: ChatCompletionRequest,
    guardrails_config: GuardrailsConfig | None = None,
) -> ChatCompletionResponse | Iterator[ChatCompletionChunk]:
    """Execute chat completion with the specified provider.

    When ``request.stream`` is true, returns an iterator of streaming
    chunks instead of a single response. Output guardrails are applied
    per chunk (log-only — no mid-stream blocking).

    Args:
        provider: Provider name ("openai", "anthropic", "google").
        api_key: API key for the provider.
        request: Chat completion request. Set ``stream=True`` for streaming.
        guardrails_config: Optional guardrails configuration.

    Returns:
        ``ChatCompletionResponse`` when ``request.stream`` is falsy, or
        ``Iterator[ChatCompletionChunk]`` when ``request.stream`` is true.

    Raises:
        GuardrailsError: If content matches a block rule.
    """
    config = guardrails_config if guardrails_config is not None else _get_automatic_config()

    if config:
        for msg in request.messages:
            if not msg.content:
                continue
            redacted, violations = check_guardrails(msg.content, config, GuardrailDirection.INPUT)
            for v in violations:
                logger.warning("GUARDRAIL %s [%s]: %s", v.action.name, v.direction.name, v)
            if redacted != msg.content:
                msg.content = redacted

    provider_instance = get_provider(provider, api_key, model=request.model)

    if request.stream:
        return _iter_stream(provider_instance.stream(request), config)

    response = provider_instance.chat_complete(request)
    if config:
        for choice in response.choices:
            if content := choice.message.content:
                redacted, violations = check_guardrails(content, config, GuardrailDirection.OUTPUT)
                for v in violations:
                    logger.warning("GUARDRAIL %s [%s]: %s", v.action.name, v.direction.name, v)
                if redacted != content:
                    choice.message.content = redacted

    return response


def _iter_stream(
    chunks: Iterator[ChatCompletionChunk], config: GuardrailsConfig | None
) -> Iterator[ChatCompletionChunk]:
    for chunk in chunks:
        if config:
            for choice in chunk.choices:
                if content := choice.delta.content:
                    for v in check_guardrails(content, config, GuardrailDirection.OUTPUT):
                        logger.warning("GUARDRAIL %s [%s]: %s", v.action.name, v.direction.name, v)
        yield chunk


def get_available_providers() -> list[Provider]:
    """Get list of available providers.

    Returns:
        List of provider names.
    """
    return list_providers()


class GuardrailsResource:
    """Server-side PII detection via the guardrail service.

    Accessed as ``gateway.guardrails.scan(text)``.
    """

    def __init__(self, gateway: "Gateway") -> None:
        self._gateway = gateway

    def scan(self, text: str) -> ScanResponse:
        """Scan text for PII entities using workspace guardrail config.

        Args:
            text: The text to scan.

        Returns:
            ScanResponse with detected entities, action taken, and processed text.

        Raises:
            RouterAPIError: If the API call fails.
        """
        return _api_scan_text(self._gateway.client, text)


class Gateway:
    """Gateway class with default provider and API key support.
    Exposes a single ``chat_complete()`` entry point that handles both
    streaming and non-streaming based on ``request.stream``.

    Attributes:
        default_provider: Default provider to use when not specified.
        default_api_key: Default API key to use when not specified.
        guardrails_config: Optional guardrails configuration for content filtering.
        server_guardrails: Enable the hosted PII scan. ``True`` scans both input
            and output; pass a ``ServerGuardrailsConfig`` to scan only one side.
        guardrails: Server-side PII detection resource.
    """

    def __init__(
        self,
        default_provider: Provider | str | None = None,
        default_api_key: str | None = None,
        guardrails_config: GuardrailsConfig | None = None,
        server_guardrails: bool | ServerGuardrailsConfig = False,
        REZUNATE_LLM_API_KEY: str | None = None,
    ) -> None:
        """Initialize Gateway.

        Args:
            default_provider: Default provider to use.
            default_api_key: Default API key to use for the LLM provider.
            guardrails_config: Optional guardrails configuration for content filtering.
            server_guardrails: When True, scan LLM output with the hosted PII
                service and block/redact it.
            REZUNATE_LLM_API_KEY: API key for the LLM-Router API (falls back to REZUNATE_LLM_API_KEY env var).
        """
        self.default_provider = default_provider
        self.default_api_key = default_api_key
        self.guardrails_config = guardrails_config
        self.server_guardrails = server_guardrails
        self._REZUNATE_LLM_API_KEY = REZUNATE_LLM_API_KEY
        self._client: RouterClient | None = None
        self.guardrails = GuardrailsResource(self)

    @property
    def client(self) -> RouterClient:
        """Lazily-created RouterClient for the LLM-Router API."""
        if self._client is None:
            kwargs: dict = {}
            if self._REZUNATE_LLM_API_KEY:
                kwargs["api_key"] = self._REZUNATE_LLM_API_KEY
            self._client = RouterClient(**kwargs)
        return self._client

    def chat_complete(
        self,
        request: ChatCompletionRequest,
        provider: str | Provider | None = None,
        api_key: str | None = None,
        guardrails_config: GuardrailsConfig | None = None,
        server_guardrails: bool | ServerGuardrailsConfig | None = None,
    ) -> ChatCompletionResponse | Iterator[ChatCompletionChunk]:
        """Execute chat completion. Streams if ``request.stream`` is true.

        Args:
            request: Chat completion request with model, messages, and
                parameters. Set ``stream=True`` to receive an iterator of
                ``ChatCompletionChunk`` objects instead of a single response.
            provider: Provider name (uses ``default_provider`` if not specified).
            api_key: API key (uses ``default_api_key`` if not specified).
            guardrails_config: Optional guardrails config.
            server_guardrails: Override the instance ``server_guardrails`` setting
                for this call. ``True`` scans both input and output with the
                hosted PII service.

        Returns:
            ``ChatCompletionResponse`` when ``request.stream`` is falsy, or
            ``Iterator[ChatCompletionChunk]`` when ``request.stream`` is true.

        Raises:
            ValueError: If provider or api_key is not specified and no default is set.
            GuardrailsError: If input or output content matches a local regex rule.
            ServerGuardrailsError: If the hosted PII scan blocks input or output.
        """
        resolved_provider = provider or self.default_provider
        resolved_api_key = api_key or self.default_api_key
        resolved_guardrails = guardrails_config or self.guardrails_config
        server = self.server_guardrails if server_guardrails is None else server_guardrails
        # True -> scan both sides; a config picks the side(s); False/None -> off.
        server_config = (
            (server if isinstance(server, ServerGuardrailsConfig) else ServerGuardrailsConfig())
            if server
            else None
        )

        if not resolved_provider:
            raise ValueError("Provider must be specified")
        if not resolved_api_key:
            raise ValueError("API key must be specified")

        if server_config and GuardrailDirection.INPUT in server_config.directions:
            for msg in request.messages:
                if msg.content:
                    msg.content = self._server_scan(msg.content, GuardrailDirection.INPUT)

        response = chat_complete(
            provider=resolved_provider,
            api_key=resolved_api_key,
            request=request,
            guardrails_config=resolved_guardrails,
        )

        if server_config and GuardrailDirection.OUTPUT in server_config.directions:
            for choice in response.choices:
                if choice.message.content:
                    choice.message.content = self._server_scan(
                        choice.message.content, GuardrailDirection.OUTPUT
                    )

        return response

    def _server_scan(self, text: str, direction: GuardrailDirection) -> str:
        """Scan one piece of text with the hosted PII service and apply its action.

        Args:
            text: The input or output text to scan.
            direction: Which side this text is, for error reporting/logging.

        Returns:
            The text, redacted by the server if PII was found, or unchanged.

        Raises:
            ServerGuardrailsError: If the scan blocks the text.
        """
        result = self.guardrails.scan(text)
        for ent in result.entities:
            logger.warning(
                "SERVER GUARDRAIL [%s]: %s %r (score=%.2f)",
                direction.name,
                ent.label,
                ent.text,
                ent.score,
            )

        if result.blocked:
            raise ServerGuardrailsError(direction, result.entities, result.action)

        if result.text and result.text != text:
            return result.text
        return text

    def get_prompt(
        self,
        slug_id: str,
        variables: dict[str, str] | None = None,
        version: int | None = None,
    ) -> str:
        """Fetch a prompt from the LLM-Router API and render it.

        Args:
            slug_id: The prompt's slug identifier.
            variables: Optional mapping of template variable names to values.
            version: Optional version number to pin to.

        Returns:
            The rendered prompt string.

        Raises:
            RouterAPIError: If api_key is not configured (and not in env).
        """
        prompt = _api_get_prompt(self.client, slug_id, version=version)
        return render_prompt(prompt.content, variables)

    @property
    def providers(self) -> list[Provider]:
        """Get available providers.

        Returns:
            List of provider names.
        """
        return list_providers()
