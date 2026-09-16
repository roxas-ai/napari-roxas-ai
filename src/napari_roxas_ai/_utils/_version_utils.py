"""
Provenance values recorded in sample metadata: which software produced a
measurement run, and who ran it.

The version itself lives in ``napari_roxas_ai.__init__.__version__``, which
``pyproject.toml`` also reads as the package version, so there is a single
source of truth.
"""

import getpass

from napari_roxas_ai import __version__

SOFTWARE_NAME = "ROXAS AI"

# Written instead of the operator name when the account name is unavailable,
# so that the key is always present and never silently missing.
UNKNOWN_OPERATOR = "unknown"


def get_software_version() -> str:
    """
    Return the software version string written to sample metadata.

    Returns
    -------
    str
        The product name followed by the package version, e.g. "ROXAS AI 0.1.2".
    """
    return f"{SOFTWARE_NAME} {__version__}"


def get_measurement_operator() -> str:
    """
    Return the name of the account running the measurement.

    Recorded as ``meas_by`` next to ``meas_created_at`` and ``sw_version``, so
    that a result can be traced back to the person who produced it.

    Returns
    -------
    str
        The operating system account name, or "unknown" if it cannot be
        determined (getpass raises when no account name is set anywhere).
    """
    try:
        operator = getpass.getuser()
    except Exception:
        return UNKNOWN_OPERATOR

    return operator.strip() or UNKNOWN_OPERATOR
