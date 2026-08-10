"""
Prefixes deciding which metadata keys survive a sample metadata round trip.

One ``.metadata.json`` is shared by a sample's scan, cells and rings layers.
Each layer only loads and only writes its own namespace, so that saving one
layer cannot overwrite another layer's keys with the stale snapshot it happened
to load. Keys matching none of the applicable prefixes are dropped silently,
so a new metadata key must be covered here to be read back or written out.
"""

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
    "adjacent_cwt_ratio_limit",
)

# Keys describing the run that produced the output rather than one content type.
# Listed explicitly rather than as a broad "meas_" prefix, because the meas_
# namespace also holds sample properties such as meas_geometry, which must not
# be rewritten by a content save.
RUN_METADATA_PREFIXES = (
    "meas_created_at",
    "sw_",
    *MEASUREMENT_PARAMETER_KEYS,
)
