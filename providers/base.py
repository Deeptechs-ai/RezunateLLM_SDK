"""
Base Provider Class.
All providers inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import requests


class BaseProvider(ABC):
    """
    Abstract base class for all providers.
    Each provider must implement these methods.
    """

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for the provider's API."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return the headers required for API calls."""
        pass

    @abstractmethod
    def get_endpoint(self) -> str:
        """Return the chat completion endpoint."""
        pass

    @abstractmethod
    def transform_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform OpenAI format request to provider's format.
        For OpenAI provider, this returns the request unchanged.
        """
        pass

    @abstractmethod
    def transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform provider's response to OpenAI format.
        For OpenAI provider, this returns the response unchanged.
        """
        pass

    def chat_complete(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute chat completion.

        1. Transform request to provider format
        2. Call provider API
        3. Transform response to OpenAI format
        4. Return response
        """
        # Transform request to provider format
        provider_request = self.transform_request(request)

        # Build full URL
        url = f"{self.base_url}{self.get_endpoint()}"

        # Make API call
        try:
            response = requests.post(
                url,
                headers=self.get_headers(),
                json=provider_request,
                timeout=60
            )
            response.raise_for_status()
            provider_response = response.json()

        except requests.exceptions.RequestException as e:
            return {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                },
                "provider": self.provider_name
            }

        # Transform response to OpenAI format
        openai_response = self.transform_response(provider_response)
        openai_response["provider"] = self.provider_name

        return openai_response
