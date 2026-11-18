"""
Prefixes deciding which metadata keys survive a sample metadata round trip.

One ``.metadata.json`` is shared by a sample's scan, cells and rings layers.
Each layer only loads and only writes its own namespace, so that saving one
layer cannot overwrite another layer's keys with the stale snapshot it happened
to load. Keys matching none of the applicable prefixes are dropped silently,
so a new metadata key must be covered here to be read back or written out.
"""

from typing import Any, Optional

import numpy as np


# Sample-level keys, meaningful for every layer of a sample. Entries without a
# trailing underscore are whole key names rather than prefixes; ``startswith``
# treats both alike. ``meas_geometry`` is listed here despite its ``meas_``
# prefix because it is a sample property set during preparation, not a property
# of a measurement run.
SAMPLE_METADATA_PREFIXES = (
    "sample_",
    "spatial_resolution",
    "meas_geometry",
    "reference_series",
)

# The stored reference series value meaning "none selected". Treated as absent
# when restoring, so that a sample saved without a reference can still be
# detected by name pattern matching later on.
NO_REFERENCE_SERIES = "NA"

# Measurement parameters recorded alongside the results so that a run can be
# reproduced. All of them describe cell wall thickness processing and are
# therefore written with the cells output only. The keys are also the config
# keys passed to SampleAnalyzer, which is what lets the widgets copy them
# straight from the config of the run.
MEASUREMENT_PARAMETER_KEYS = (
    "cluster_dbl_cwt_threshold",
    "relwidth_cwt_integration",
    "lower_limit_cwt_iqr_multiplier",
    "upper_limit_cwt_iqr_multiplier",
    "opposite_cwt_ratio_limit",
    "opposite_cwt_percentile_exception",
    "adjacent_cwt_ratio_limit",
)

# Keys describing the run that produced the output rather than one content type.
# Listed explicitly rather than as a broad "meas_" prefix, because the meas_
# namespace also holds sample properties such as meas_geometry, which must not
# be rewritten by a content save.
RUN_METADATA_PREFIXES = (
    "meas_created_at",
    "meas_by",
    "sw_",
    *MEASUREMENT_PARAMETER_KEYS,
)

# Metadata keys that earlier ROXAS AI versions wrote under a different name.
# Samples prepared with those versions are still read back, so every entry here
# is renamed on load and on save; the file is rewritten under the current name
# the next time the sample is saved. Keep old names in this map forever, since
# a project directory can hold samples from any past version.
LEGACY_METADATA_KEY_RENAMES = {
    "sample_geometry": "meas_geometry",
    "sample_scale": "spatial_resolution",
}


def migrate_legacy_metadata_keys(metadata: dict) -> dict:
    """
    Rename legacy metadata keys to their current names.

    The renamed key keeps the position of the legacy key, so a migrated file
    keeps the field order of a freshly written one. If both the legacy and the
    current name are present, the current one wins and the legacy one is
    dropped.

    Parameters
    ----------
    metadata : dict
        Metadata as read from a ``.metadata.json`` file.

    Returns
    -------
    dict
        A new dict with legacy keys renamed. Non-dict input is returned as-is.
    """
    if not isinstance(metadata, dict):
        return metadata

    if not any(key in metadata for key in LEGACY_METADATA_KEY_RENAMES):
        return metadata

    migrated = {}
    for key, value in metadata.items():
        new_key = LEGACY_METADATA_KEY_RENAMES.get(key, key)
        if new_key != key and new_key in metadata:
            # The current key is already there and takes precedence
            continue
        migrated[new_key] = value
    return migrated


# Specific precision (decimal places) for float metadata keys.
# Keys not listed here fall back to DEFAULT_METADATA_FLOAT_DECIMALS.
METADATA_FLOAT_PRECISION = {
    "spatial_resolution": 6,
    "sample_angle": 2,
    "resolution_dpi": 2,
    "cluster_dbl_cwt_threshold": 2,
    "relwidth_cwt_integration": 2,
    "lower_limit_cwt_iqr_multiplier": 2,
    "upper_limit_cwt_iqr_multiplier": 2,
    "opposite_cwt_ratio_limit": 2,
    "opposite_cwt_percentile_exception": 2,
    "adjacent_cwt_ratio_limit": 2,
}

DEFAULT_METADATA_FLOAT_DECIMALS = 4


def sanitize_metadata_floats(
    obj: Any,
    default_decimals: int = DEFAULT_METADATA_FLOAT_DECIMALS,
    key_name: Optional[str] = None,
) -> Any:
    """
    Recursively round float values in metadata to prevent IEEE 754 precision artifacts.

    Parameters
    ----------
    obj : Any
        The metadata object (dict, list, float, etc.) to sanitize.
    default_decimals : int, default 4
        Fallback decimal places for floats not matching a specific key rule.
    key_name : str, optional
        The key name of the current field if traversing a dict.

    Returns
    -------
    Any
        A copy of the object with all float values rounded.
    """
    if isinstance(obj, (float, np.floating)):
        decimals = METADATA_FLOAT_PRECISION.get(key_name, default_decimals)
        return round(float(obj), decimals)
    if isinstance(obj, dict):
        return {
            k: sanitize_metadata_floats(v, default_decimals, key_name=str(k))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        res = [
            sanitize_metadata_floats(item, default_decimals, key_name=key_name)
            for item in obj
        ]
        return tuple(res) if isinstance(obj, tuple) else res
    return obj
