import os
import napari
import numpy as np
from napari.settings import get_settings
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

# Track which viewer instances already had shortcuts applied
_APPLIED_VIEWERS = set()


def has_shortcuts_applied(viewer) -> bool:
    """Return True if shortcuts were already installed for this viewer."""
    return id(viewer) in _APPLIED_VIEWERS


def mark_shortcuts_applied(viewer) -> None:
    """Record that shortcuts were installed for this viewer."""
    _APPLIED_VIEWERS.add(id(viewer))


def close_all_dock_widgets(viewer=None) -> None:
    """Close all open dock widgets in the napari viewer."""
    if viewer is None:
        try:
            viewer = napari.current_viewer()
        except Exception:
            viewer = None

    if viewer is None or not hasattr(viewer, "window") or viewer.window is None:
        return

    for name in list(viewer.window.dock_widgets.keys()):
        w = viewer.window.dock_widgets.get(name)
        if w is None:
            continue
        try:
            viewer.window.remove_dock_widget(w)
        except Exception:
            pass
        p = getattr(w, "native", w)
        while p is not None:
            if hasattr(p, "close"):
                try:
                    viewer.window.remove_dock_widget(p)
                except Exception:
                    p.close()
                break
            p = p.parent() if hasattr(p, "parent") else None


def _build_group_scroll_container(viewer, widget_items) -> QScrollArea:
    """Create a unified, scrollable container hosting multiple widgets with titles aligned at top."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(6)
    layout.setAlignment(Qt.AlignTop)

    hosted_classes = []
    for idx, (title, cls) in enumerate(widget_items):
        if idx > 0:
            layout.addSpacing(10)

        header = QLabel(f"<b>{title}</b>")
        header.setStyleSheet(
            "QLabel { font-size: 11pt; padding: 4px 2px; border-bottom: 1px solid rgba(128, 128, 128, 0.4); }"
        )
        layout.addWidget(header)

        w = cls(viewer)
        native = getattr(w, "native", w)
        layout.addWidget(native)
        hosted_classes.append(cls)

    container._hosted_widget_classes = hosted_classes
    scroll.setWidget(container)
    return scroll


def is_widget_already_present(viewer, widget_cls) -> bool:
    """Check if a widget of class widget_cls is already present (individually or inside a group)."""
    if viewer is None or not hasattr(viewer, "window") or viewer.window is None:
        return False

    for name, dock in viewer.window.dock_widgets.items():
        w = getattr(dock, "widget", lambda: dock)()
        if isinstance(w, widget_cls):
            return True
        if hasattr(w, "_hosted_widget_classes") and widget_cls in w._hosted_widget_classes:
            return True
        if isinstance(w, QScrollArea) and w.widget() is not None:
            inner = w.widget()
            if hasattr(inner, "_hosted_widget_classes") and widget_cls in inner._hosted_widget_classes:
                return True
            for child in inner.findChildren(QWidget):
                if isinstance(child, widget_cls):
                    return True

    return False


def prevent_duplicate_dock_if_already_present(widget, viewer, title: str) -> bool:
    """
    If widget's class is already open in viewer (individually or in a group),
    notifies user and automatically removes this duplicate dock widget.
    Returns True if a duplicate was detected and scheduled for removal.
    """
    if is_widget_already_present(viewer, widget.__class__):
        from napari.utils.notifications import show_info
        from qtpy.QtCore import QTimer

        show_info(f"'{title}' is already active in the sidebar.")

        def _remove_duplicate():
            p = getattr(widget, "native", widget)
            while p is not None:
                if hasattr(p, "parent") and p.parent() is not None:
                    p = p.parent()
                    if hasattr(viewer, "window") and hasattr(viewer.window, "remove_dock_widget"):
                        try:
                            viewer.window.remove_dock_widget(p)
                            break
                        except Exception:
                            pass
                else:
                    break

        QTimer.singleShot(0, _remove_duplicate)
        return True
    return False


def create_manual_editing_group_widget(viewer=None):
    """Return the unified Manual Editing group widget."""
    if viewer is None:
        try:
            viewer = napari.current_viewer()
        except Exception:
            viewer = None

    from napari_roxas_ai._edition._cells_layer_editor import CellsLayerEditorWidget
    from napari_roxas_ai._edition._rings_layer_editor import RingsLayerEditorWidget
    from napari_roxas_ai._loading._samples_loading_widget import SamplesLoadingWidget
    from napari_roxas_ai._measurements._single_sample_measurements import (
        SingleSampleMeasurementsWidget,
    )
    from napari_roxas_ai._segmentation._single_sample_segmentation import (
        SingleSampleSegmentationWidget,
    )

    items = [
        ("2/4 - Load image(s)", SamplesLoadingWidget),
        (
            "3A - Detect cells & rings (individual image)",
            SingleSampleSegmentationWidget,
        ),
        ("5 - Edit cells", CellsLayerEditorWidget),
        ("6 - Edit rings", RingsLayerEditorWidget),
        (
            "9A - Measure cells & rings (individual image)",
            SingleSampleMeasurementsWidget,
        ),
    ]
    return _build_group_scroll_container(viewer, items)


def create_batch_operations_group_widget(viewer=None):
    """Return the unified Batch Operations group widget."""
    if viewer is None:
        try:
            viewer = napari.current_viewer()
        except Exception:
            viewer = None

    from napari_roxas_ai._measurements._batch_sample_measurements import (
        BatchSampleMeasurementsWidget,
    )
    from napari_roxas_ai._preparation._preparation_widget import PreparationWidget
    from napari_roxas_ai._segmentation._batch_sample_segmentation import (
        BatchSampleSegmentationWidget,
    )

    items = [
        ("1 - Prepare project images for analysis", PreparationWidget),
        ("3B - Batch detect cells & rings", BatchSampleSegmentationWidget),
        ("9B - Batch measure & update", BatchSampleMeasurementsWidget),
    ]
    return _build_group_scroll_container(viewer, items)


def load_manual_editing_group(viewer=None):
    """Return the Manual Editing unified widget group container."""
    if viewer is None:
        try:
            viewer = napari.current_viewer()
        except Exception:
            viewer = None

    if viewer is not None and not has_shortcuts_applied(viewer):
        install_wasd_shortcuts(viewer)

    if viewer is not None:
        close_all_dock_widgets(viewer)

    group_widget = create_manual_editing_group_widget(viewer)
    return group_widget


def load_batch_operations_group(viewer=None):
    """Return the Batch Operations unified widget group container."""
    if viewer is None:
        try:
            viewer = napari.current_viewer()
        except Exception:
            viewer = None

    if viewer is not None and not has_shortcuts_applied(viewer):
        install_wasd_shortcuts(viewer)

    if viewer is not None:
        close_all_dock_widgets(viewer)

    group_widget = create_batch_operations_group_widget(viewer)
    return group_widget


def install_wasd_shortcuts(viewer):
    """
    Install WASD pan + Q/E zoom and widget group shortcuts on the given napari viewer.
    Removes default Q/W/E/A/S/D shortcuts first.
    """
    mark_shortcuts_applied(viewer)

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
    # Accessing qt_viewer is deprecated, so we attempt to safely refresh
    # using getattr to avoid warnings and handle version differences.
    qtv = getattr(viewer.window, "_qt_viewer", None)
    if qtv is None:
        qtv = getattr(viewer.window, "qt_viewer", None)

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

    # --------------------------
    # WIDGET GROUP SHORTCUTS
    # --------------------------
    def _dock_manual_group(v):
        w = load_manual_editing_group(v)
        if v is not None and hasattr(v, "window") and v.window is not None:
            v.window.add_dock_widget(w, area="right", name="Group - Manual Editing")

    def _dock_batch_group(v):
        w = load_batch_operations_group(v)
        if v is not None and hasattr(v, "window") and v.window is not None:
            v.window.add_dock_widget(w, area="right", name="Group - Batch Operations")

    group_shortcuts = {
        "Control-Shift-M": lambda v, *a: _dock_manual_group(v),
        "Control-Shift-B": lambda v, *a: _dock_batch_group(v),
        "Control-Shift-W": lambda v, *a: close_all_dock_widgets(v),
    }
    for key, cb in group_shortcuts.items():
        viewer.bind_key(key, cb, overwrite=True)

    print(f"WASD and arrow shortcuts installed ({removed} default binds removed).")
