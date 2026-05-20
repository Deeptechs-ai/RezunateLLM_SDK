"""LLM-Router API endpoints."""

from rezunate_llm_sdk.client import RouterClient
from rezunate_llm_sdk.models import PromptResponse, ScanResponse

API_VERSION = "v1"
PROMPT_ENDPOINT = f"/api/{API_VERSION}/prompts"
GUARDRAILS_ENDPOINT = f"/api/{API_VERSION}/guardrails"

# Scans can take a while when the inference service cold-starts
SCAN_TIMEOUT_SECONDS = 180


def get_prompt(client: RouterClient, slug_id: str, version: int | None = None) -> PromptResponse:
    """Fetch a prompt by slug from the LLM-Router API.

    Args:
        client: Authenticated RouterClient instance.
        slug_id: The prompt's slug identifier.
        version: Optional version number to pin to.

    Returns:
        PromptResponse with the prompt data.

    Raises:
        RouterAPIError: If the API returns an error or the request fails.
    """
    params = {}
    if version is not None:
        params["version"] = version
    resp = client.request("GET", f"{PROMPT_ENDPOINT}/{slug_id}", params=params)
    return PromptResponse.model_validate(resp.json())


def scan_text(client: RouterClient, text: str) -> ScanResponse:
    """Scan text for PII entities using the server-side guardrail service.

    Args:
        client: Authenticated RouterClient instance.
        text: The text to scan for PII.

    Returns:
        ScanResponse with detected entities, action, and processed text.

    Raises:
        RouterAPIError: If the API returns an error or the request fails.
    """
    resp = client.request(
        "POST",
        f"{GUARDRAILS_ENDPOINT}/scan",
        json={"text": text},
        timeout=SCAN_TIMEOUT_SECONDS,
    )
    return ScanResponse.model_validate(resp.json())
