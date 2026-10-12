import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import requests

REPO = "roxas-ai/napari-roxas-ai"

# Stores the GitHub id of the release asset a directory was last filled from
ASSET_ID_FILE = ".asset_id"


def get_asset(asset_name: str) -> dict:
    """
    Get the latest asset file from the GitHub repository.
    Args:
        asset_name (str): The name of the asset file to download.
    Returns:
        dict: The GitHub asset (with "id" and "browser_download_url").
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

    resp = None
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
        if resp is not None:
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
            return asset
    raise ValueError(f"Asset '{asset_name}' not found in the latest release.")


def get_asset_file_url(asset_name: str) -> str:
    """
    Get the URL of the latest asset file from the GitHub repository.
    Args:
        asset_name (str): The name of the asset file to download.
    Returns:
        str: The URL of the asset file.
    """
    return get_asset(asset_name)["browser_download_url"]


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

    file_name = url.split("/")[-1]

    # Work in a temporary directory, so that a failed download or extraction
    # leaves neither the archive nor partial content in the destination
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        file_path = temp_path / file_name

        with open(file_path, "wb") as f:
            f.write(response.content)

        if file_name.endswith(".zip"):
            extract_path = temp_path / "extracted"
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Get the first directory in the archive (our root folder)
            root_dir = next(extract_path.iterdir())
            items = list(root_dir.iterdir())
        else:
            items = [file_path]

        # Move the content to the destination, replacing files that already
        # exist there (e.g. models kept from a previous release)
        for item in items:
            target = dest_path / item.name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(item), str(target))


def _read_asset_id(directory_path: Path) -> Optional[str]:
    """Read the id of the release asset the directory was last filled from."""
    try:
        return (directory_path / ASSET_ID_FILE).read_text().strip() or None
    except OSError:
        return None


def check_assets_and_download(
    directory: str, asset_name: str, required_files: tuple = ()
) -> None:
    """
    Check if the directory contains the assets, and download them if not.

    The assets are downloaded if the directory does not exist, is empty (left
    behind by an interrupted download), or lacks one of the required files
    (e.g. a model added in a newer release). In the latter case nothing is
    downloaded again if the latest release asset was already downloaded, so a
    release that lacks the file does not trigger a download on every call.
    If the download fails while assets are already present, they are kept
    and used (e.g. when working offline).

    Args:
        directory (str): The directory where the assets are stored.
        asset_name (str): The name of the asset file to download.
        required_files (tuple): File names that must be present in the directory.
    """

    directory_path = Path(directory)

    # Remove an archive left behind by a failed download of version 0.2.0
    stale_archive = directory_path / asset_name
    if stale_archive.is_file():
        stale_archive.unlink()

    has_files = directory_path.exists() and any(directory_path.iterdir())
    missing_files = [
        name for name in required_files if not (directory_path / name).exists()
    ]
    if has_files and not missing_files:
        return

    try:
        asset = get_asset(asset_name)
    except (ConnectionError, ValueError) as e:
        if not has_files:
            raise
        print(
            f"WARNING: Could not check for missing assets {missing_files} "
            f"in {asset_name} ({e}). Using the assets present in {directory}."
        )
        return

    asset_id = str(asset["id"])
    if has_files and _read_asset_id(directory_path) == asset_id:
        print(
            f"WARNING: The latest {asset_name} does not contain {missing_files}."
        )
        return

    print(
        f"Directory {directory} is missing assets. Downloading assets in {asset_name}..."
    )
    download_and_decompress_file(asset["browser_download_url"], directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    (directory_path / ASSET_ID_FILE).write_text(asset_id)
    print(f"Assets from {asset_name} have been downloaded in {directory}")
