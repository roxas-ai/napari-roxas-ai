"""Module for managing callbacks across different widgets."""

from weakref import WeakKeyDictionary

# Global dictionary to store callbacks, using weak references to layers
# This prevents memory leaks if layers are deleted
_layer_callbacks: WeakKeyDictionary = WeakKeyDictionary()


def register_layer_callback(layer, callback_owner, callback_function):
    """
    Register a callback for a layer from a specific owner.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer to register the callback for
    callback_owner : object
        The widget or object that owns this callback
    callback_function : callable
        The function to call when the layer changes

    Returns
    -------
    function
        The connected callback reference that can be used for disconnection
    """
    # Initialize entry for this layer if it doesn't exist
    if layer not in _layer_callbacks:
        _layer_callbacks[layer] = {}

    # Connect the callback to the layer's events
    callback_ref = layer.events.connect(callback_function)

    # Store the callback reference
    _layer_callbacks[layer][callback_owner] = callback_ref

    return callback_ref


def unregister_layer_callback(layer, callback_owner):
    """
    Unregister a callback for a layer from a specific owner.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer to unregister the callback from
    callback_owner : object
        The widget or object that owns the callback
    """
    # Check if we have callbacks for this layer and if owner has a callback
    if layer in _layer_callbacks and callback_owner in _layer_callbacks[layer]:

        # Disconnect the callback
        # The callback is automatically removed when disconnected
        del _layer_callbacks[layer][callback_owner]

        # If no more callbacks for this layer, remove the layer entry
        if not _layer_callbacks[layer]:
            del _layer_callbacks[layer]


def get_layer_callbacks(layer):
    """
    Get all callbacks registered for a layer.

    Parameters
    ----------
    layer : napari.layers.Layer
        The layer to get callbacks for

    Returns
    -------
    dict
        Dictionary of callbacks by owner
    """
    if layer in _layer_callbacks:
        return _layer_callbacks[layer]
    return {}
