"""LLM-Router API endpoints."""

from llm_router.client import RouterClient
from llm_router.models import PromptResponse

API_VERSION = "v1"
PROMPT_ENDPOINT = f"/api/{API_VERSION}/prompts"


def get_prompt(client: RouterClient, slug_id: str) -> PromptResponse:
    """Fetch a prompt by slug from the LLM-Router API.

    Args:
        client: Authenticated RouterClient instance.
        slug_id: The prompt's slug identifier.

    Returns:
        PromptResponse with the prompt data.

    Raises:
        RouterAPIError: If the API returns an error or the request fails.
    """
    resp = client.request("GET", f"{PROMPT_ENDPOINT}/{slug_id}")
    return PromptResponse.model_validate(resp.json())
