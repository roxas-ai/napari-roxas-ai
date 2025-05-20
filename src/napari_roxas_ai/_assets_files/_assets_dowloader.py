import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

REPO = "roxas-ai/napari-roxas-ai"


def get_asset_file_url(asset_name: str) -> str:
    """
    Get the URL of the latest asset file from the GitHub repository.
    Args:
        asset_name (str): The name of the asset file to download.
    Returns:
        str: The URL of the asset file.
    Raises:
        ValueError: If the asset file is not found in the latest release.
    """

    url = f"https://api.github.com/repos/{REPO}/releases"

    # Set up headers with authentication token if available
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
        print("Using GitHub token for authentication")

    try:
        print(f"Requesting GitHub API at {url}")
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.HTTPError,
    ) as e:
        print(
            "WARNING: Unable to connect to GitHub API. No internet connection or GitHub API is unavailable."
        )
        print(f"Error: {e}")
        print(
            f"Rate limit headers: {resp.headers.get('X-RateLimit-Limit')}, Remaining: {resp.headers.get('X-RateLimit-Remaining')}"
        )
        raise ConnectionError(
            "Cannot access GitHub releases. Please check your internet connection and try again."
        ) from e

    releases = resp.json()

    if not releases:
        raise ValueError("No releases found.")

    # Optionally filter out drafts if needed
    latest = next(r for r in releases if not r["draft"])

    for asset in latest.get("assets", []):
        if asset["name"] == asset_name:
            return asset["browser_download_url"]
    raise ValueError(f"Asset '{asset_name}' not found in the latest release.")


def download_and_decompress_file(url: str, dest: str) -> None:
    """
    Downloads a file from a given URL.
    Then decompresses the file.
    Then puts the content of the decompressed file in the destination directory.

    Args:
        url (str): The URL of the file to download.
        dest (str): The destination directory where the file will be saved.
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Set up headers with authentication token if available
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    # Download the file
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Save the file to the destination directory
    file_name = url.split("/")[-1]
    file_path = dest_path / file_name

    with open(file_path, "wb") as f:
        f.write(response.content)

    # Decompress the file
    if file_name.endswith(".zip"):

        # Create a temporary directory for initial extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract to temp directory
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(Path(temp_dir))

            # Get the first directory in temp_dir (our root folder)
            root_dir = next(Path(temp_dir).iterdir())

            # Move contents from temp/root_dir to destination
            for item in root_dir.iterdir():
                shutil.move(str(item), str(dest_path))

        # Remove the zip file
        file_path.unlink()


def check_assets_and_download(directory: str, asset_name: str) -> None:
    """
    Check if the directory exits
    If not, download and decompress assets in their directory.

    Args:
        directory (str): The directory where the assets are stored.
        asset_name (str): The name of the asset file to download.
    """

    if not Path(directory).exists():

        print(
            f"Directory {directory} does not exist. Downloading assets in {asset_name}..."
        )
        url = get_asset_file_url(asset_name)
        download_and_decompress_file(url, directory)
        print(f"Assets from {asset_name} have been downloaded in {directory}")
