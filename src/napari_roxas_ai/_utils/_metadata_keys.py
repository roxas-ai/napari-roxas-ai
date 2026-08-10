"""
Prefixes deciding which metadata keys survive a sample metadata round trip.

One ``.metadata.json`` is shared by a sample's scan, cells and rings layers.
Each layer only loads and only writes its own namespace, so that saving one
layer cannot overwrite another layer's keys with the stale snapshot it happened
to load. Keys matching none of the applicable prefixes are dropped silently,
so a new metadata key must be covered here to be read back or written out.
"""

# Sample-level keys, meaningful for every layer of a sample.
SAMPLE_METADATA_PREFIXES = ("sample_", "spatial_resolution")

# Keys describing the run that produced the output rather than one content type.
RUN_METADATA_PREFIXES = ("meas_", "sw_")
