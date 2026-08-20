import numpy as np
from napari.settings import get_settings

# Track which viewer instances already had shortcuts applied
_APPLIED_VIEWERS = set()


def has_shortcuts_applied(viewer) -> bool:
    """Return True if shortcuts were already installed for this viewer."""
    return id(viewer) in _APPLIED_VIEWERS


def mark_shortcuts_applied(viewer) -> None:
    """Record that shortcuts were installed for this viewer."""
    _APPLIED_VIEWERS.add(id(viewer))


def install_wasd_shortcuts(viewer):
    """
    Install WASD pan + Q/E zoom shortcuts on the given napari viewer.
    Removes default Q/W/E/A/S/D shortcuts first.
    """

    SETTINGS = get_settings()

    # Backup current shortcuts globally once
    if "SHORTCUTS_BACKUP" not in globals():
        global SHORTCUTS_BACKUP
        SHORTCUTS_BACKUP = {
            action: list(bindings)
            for action, bindings in SETTINGS.shortcuts.shortcuts.items()
        }

    BAD = set("QWEASD")

    def _base(binding):
        """Extract single key from a shortcut string or object."""
        if isinstance(binding, str):
            s = binding
        else:
            # Some bindings are objects; try different property names
            for attr in ("key", "primary", "shortcut"):
                v = getattr(binding, attr, None)
                if v:
                    s = v
                    break
            else:
                s = str(binding)

        return str(s).split("-")[-1].upper()

    # Remove old Q/W/E/A/S/D shortcuts
    newmap = {}
    removed = 0

    for action, bindings in SETTINGS.shortcuts.shortcuts.items():
        keep = []
        for b in bindings:
            if _base(b) in BAD:
                removed += 1
            else:
                keep.append(b)
        newmap[action] = keep

    SETTINGS.shortcuts.shortcuts = newmap

    # Refresh napari internal bindings
    # Accessing _qt_viewer is deprecated, so we attempt to safely refresh
    # using available public-compatible properties.
    qtv = getattr(viewer.window, "_qt_viewer", None)
    if not qtv and hasattr(viewer.window, "qt_viewer"):
        try:
            qtv = viewer.window.qt_viewer
        except Exception:
            pass

    if qtv:
        for method in ("_bind_shortcuts", "_refresh_shortcuts", "_rebuild_shortcuts"):
            if hasattr(qtv, method):
                try:
                    getattr(qtv, method)()
                except Exception:
                    pass

    # --------------------------
    # WASD PAN MOVEMENT
    # --------------------------

    STEP, FAST = 200.0, 400.0

    def _pan_by(v, dx=0.0, dy=0.0):
        z = float(getattr(v.camera, "zoom", 1.0)) or 1.0
        c = np.array(v.camera.center, float)
        c[-2] += dy / z
        c[-1] += dx / z
        v.camera.center = tuple(c)

    bindings = {
        "w": (0, -STEP),
        "a": (-STEP, 0),
        "s": (0, STEP),
        "d": (STEP, 0),
        "Shift-w": (0, -FAST),
        "Shift-a": (-FAST, 0),
        "Shift-s": (0, FAST),
        "Shift-d": (FAST, 0),
        "Up": (0, -STEP),
        "Left": (-STEP, 0),
        "Down": (0, STEP),
        "Right": (STEP, 0),
        "Shift-Up": (0, -FAST),
        "Shift-Left": (-FAST, 0),
        "Shift-Down": (0, FAST),
        "Shift-Right": (FAST, 0),
    }

    for key, (dx, dy) in bindings.items():
        viewer.bind_key(
            key,
            lambda v, *a, dx=dx, dy=dy: _pan_by(v, dx, dy),
            overwrite=True,
        )

    # --------------------------
    # Q/E ZOOM
    # --------------------------
    def _zoom(v, step):
        v.camera.zoom = float(getattr(v.camera, "zoom", 1.0)) + step

    for key, step in {"q": -0.25, "e": 0.25}.items():
        viewer.bind_key(
            key,
            lambda v, *a, step=step: _zoom(v, step),
            overwrite=True,
        )

    print(f"WASD and arrow shortcuts installed ({removed} default binds removed).")
