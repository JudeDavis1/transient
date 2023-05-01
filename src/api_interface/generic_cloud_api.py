import requests
from dotenv import find_dotenv, load_dotenv

from .types import CloudResponse


class GenericCloudAPI:
    """
    Generic API Interface that defines a blueprint of APIs for managing cloud providers.

    This API functionality is required for:
    - Managing cloud resources
    - Training
    - Inference
    - Deployment
    - Monitoring
    """

    @property
    def base_url(self) -> str:
        """Get the base URL of the API"""
        raise NotImplementedError()

    @property
    def api_key(self) -> str:
        """Get the API key"""
        raise NotImplementedError()

    def _call_endpoint(self, method, endpoint, data: dict = {}) -> CloudResponse:
        """Call an endpoint of the API"""

        headers = {"X-Api-Key": self.api_key}
        response = requests.request(
            method=method,
            url=self.base_url + endpoint,
            params={**data, "api_key": self.api_key},
            headers=headers,
        )

        return response.json()

    def load_api_keys(self, name=".env.secret"):
        """Load environment variables"""

        load_dotenv(find_dotenv(name))
