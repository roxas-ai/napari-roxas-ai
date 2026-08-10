"""
Software version string recorded in sample metadata.

The version itself lives in ``napari_roxas_ai.__init__.__version__``, which
``pyproject.toml`` also reads as the package version, so there is a single
source of truth.
"""

from napari_roxas_ai import __version__

SOFTWARE_NAME = "ROXAS AI"


def get_software_version() -> str:
    """
    Return the software version string written to sample metadata.

    Returns
    -------
    str
        The product name followed by the package version, e.g. "ROXAS AI 0.1.2".
    """
    return f"{SOFTWARE_NAME} {__version__}"
