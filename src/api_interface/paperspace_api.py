import os

from src.api_interface.generic_cloud_api import GenericCloudAPI


class PaperspaceAPI(GenericCloudAPI):
    def __init__(self):
        # super.__init__()
        self.load_api_keys()

    def get_free_instances(self):
        """Get a list of free instances"""

        return self._call_endpoint(
            method="GET",
            endpoint="/notebooks/getNotebooks?filter={}",
        )

    def start_instance(self, instance_id):
        """Start an instance"""

        return self._call_endpoint(
            method="POST",
            endpoint=f"/machines/psfj3c701/start",
            data={
                "machineId": instance_id,
            },
        )

    @property
    def base_url(self) -> str:
        """Get the base URL of the API"""
        return "https://api.paperspace.io"

    @property
    def api_key(self) -> str:
        """Get the API key"""
        return os.getenv("PAPERSPACE_API_KEY")


if __name__ == "__main__":
    paperspace = PaperspaceAPI()
    print(paperspace.api_key)
    notebooks = paperspace.get_free_instances()["notebookList"]

    for n in notebooks:
        print(n)
