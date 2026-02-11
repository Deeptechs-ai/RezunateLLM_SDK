"""Gateway - Routes chat completion requests to providers."""

from llm_router.guardrails import check_guardrails
from llm_router.models import ChatCompletionRequest, ChatCompletionResponse, GuardrailsConfig
from llm_router.providers import get_provider, list_providers


def chat_complete(
    provider: str,
    api_key: str,
    request: ChatCompletionRequest,
    guardrails_config: GuardrailsConfig | None = None,
) -> ChatCompletionResponse:
    """Execute chat completion with the specified provider.

    Args:
        provider: Provider name ("openai", "anthropic", "google").
        api_key: API key for the provider.
        request: Chat completion request with model, messages, and parameters.
        guardrails_config: Optional guardrails configuration for content filtering.

    Returns:
        Chat completion response in OpenAI format.

    Raises:
        GuardrailsError: If input or output content matches a guardrail rule.
    """
    if guardrails_config:
        for message in request.messages:
            check_guardrails(message.content, guardrails_config, direction="input")

    provider_instance = get_provider(provider, api_key, model=request.model)
    response_dict = provider_instance.chat_complete(request.model_dump(exclude_none=True))
    response = ChatCompletionResponse.model_validate(response_dict)

    if guardrails_config:
        for choice in response.choices:
            if choice.message.content:
                check_guardrails(choice.message.content, guardrails_config, direction="output")

    return response


def get_available_providers() -> list[str]:
    """Get list of available providers.

    Returns:
        List of provider names.
    """
    return list_providers()


class Gateway:
    """Gateway class with default provider and API key support.

    Attributes:
        default_provider: Default provider to use when not specified.
        default_api_key: Default API key to use when not specified.
        guardrails_config: Optional guardrails configuration for content filtering.
    """

    def __init__(
        self,
        default_provider: str | None = None,
        default_api_key: str | None = None,
        guardrails_config: GuardrailsConfig | None = None,
    ) -> None:
        """Initialize Gateway.

        Args:
            default_provider: Default provider to use.
            default_api_key: Default API key to use.
            guardrails_config: Optional guardrails configuration for content filtering.
        """
        self.default_provider = default_provider
        self.default_api_key = default_api_key
        self.guardrails_config = guardrails_config

    def chat_complete(
        self,
        request: ChatCompletionRequest,
        provider: str | None = None,
        api_key: str | None = None,
        guardrails_config: GuardrailsConfig | None = None,
    ) -> ChatCompletionResponse:
        """Execute chat completion.

        Args:
            request: Chat completion request with model, messages, and parameters.
            provider: Provider name (uses default if not specified).
            api_key: API key (uses default if not specified).
            guardrails_config: Optional guardrails config (uses instance default if not specified).

        Returns:
            Chat completion response in OpenAI format.

        Raises:
            ValueError: If provider or api_key is not specified and no default is set.
            GuardrailsError: If input or output content matches a guardrail rule.
        """
        resolved_provider = provider or self.default_provider
        resolved_api_key = api_key or self.default_api_key
        resolved_guardrails = guardrails_config or self.guardrails_config

        if not resolved_provider:
            raise ValueError("Provider must be specified")
        if not resolved_api_key:
            raise ValueError("API key must be specified")

        return chat_complete(
            provider=resolved_provider,
            api_key=resolved_api_key,
            request=request,
            guardrails_config=resolved_guardrails,
        )

    @property
    def providers(self) -> list[str]:
        """Get available providers.

        Returns:
            List of provider names.
        """
        return get_available_providers()
