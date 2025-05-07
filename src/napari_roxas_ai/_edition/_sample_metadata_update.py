import napari


def update_sample_metadata(
    viewer: napari.Viewer,
    up_to_date_layer: napari.layers.Layer,
) -> None:
    """
    Update the sample metadata to match that of the specified layer.
    """
    # Get sample name
    sample_name = up_to_date_layer.metadata["sample_name"]
    sample_fields = [
        field
        for field in up_to_date_layer.metadata
        if field.startswith("sample_")
    ]
    new_sample_metadata = {
        field: up_to_date_layer.metadata[field]
        for field in sample_fields
        if field != "sample_name"
    }

    layers_to_update = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("sample_name") == sample_name
        and layer != up_to_date_layer
    ]

    for layer in layers_to_update:
        layer.metadata.update(new_sample_metadata)
