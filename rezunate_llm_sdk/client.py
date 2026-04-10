"""RouterClient — centralized HTTP transport for the LLM-Router API."""

import logging
import os

import requests

import rezunate_llm_sdk.constants as constants

logger = logging.getLogger(__name__)


class RouterAPIError(Exception):
    """Raised when an LLM-Router API call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RouterClient:
    """HTTP client for the LLM-Router API.

    Handles authentication, request dispatch, and error wrapping.
    Endpoint-specific methods live in ``rezunate_llm_sdk.api``.

    Args:
        api_key: API key for authentication (sent as X-API-Key header).
            Falls back to the ``ROUTER_API_KEY`` environment variable.
        timeout: Request timeout in seconds (default 30).
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("ROUTER_API_KEY", "")
        self.timeout = timeout

        if not self.api_key:
            raise RouterAPIError("api_key is required (or set ROUTER_API_KEY env var)")

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Send an authenticated request to the LLM-Router API.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API path (e.g. "/api/v1/prompts/my-slug").
            **kwargs: Extra keyword arguments forwarded to ``requests.request``.

        Returns:
            The ``requests.Response`` object.

        Raises:
            RouterAPIError: On connection or HTTP errors.
        """
        url = f"{constants.ROUTER_BASE_URL}{path}"
        headers = {"X-API-Key": self.api_key, **kwargs.pop("headers", {})}

        try:
            resp = requests.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise RouterAPIError(
                f"Cannot connect to LLM-Router API at {constants.ROUTER_BASE_URL}"
            ) from exc
        except requests.HTTPError as exc:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
            raise RouterAPIError(detail, status_code=resp.status_code) from exc

        return resp
