"""
Performance monkeypatches for napari's Shapes layer (napari 0.7.1 only).

napari's Shapes layer rebuilds its display bookkeeping for *all* shapes on
every mouse move of a drag, and uploads *all* shapes to the GPU, which makes
editing layers with tens of thousands of cells lag. These patches use napari
internals on purpose: they are applied only for the exact napari version they
were written against, and are meant to be contributed upstream.

1. Bookkeeping (``apply_shapes_patches``, exact, all Shapes layers):
   - vectorised triangle/vertex -> shape maps (``np.repeat`` instead of a
     Python loop over every shape);
   - vectorised z-order, skipped when z indices and triangle layout did not
     change (a move/resize/vertex drag never changes them);
   - ``_update_displayed`` reuses the cached gather indices when nothing
     structural changed (only the arrays the GPU upload reads are re-gathered);
   - a selection move does one display rebuild, not one per selected shape;
   - layers without non-displayed dims (2D) skip the per-shape slice test;
   - the layer extent (recomputed on every refresh) is vectorised;
   - a polygon being drawn gets a pre-padded slot, so adding vertices does
     not reallocate the whole layer's arrays and rebuild the z order.

2. View culling (``enable_view_culling(layer)``, opt-in per layer): only the
   shapes whose bounding box meets the camera view (plus a margin) are
   displayed, hit-tested and uploaded to the GPU. The visible set is only
   recomputed when the view leaves the margin.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

PATCHED_NAPARI_VERSION = "0.7.1"

# Culled area: the view extended by this fraction of its size on each side
CULL_MARGIN = 0.5

_applied = False


def apply_shapes_patches() -> bool:
    """Apply the Shapes patches once. Returns whether they are active."""
    global _applied
    if _applied:
        return True
    import napari

    if napari.__version__ != PATCHED_NAPARI_VERSION:
        logger.info(
            "napari %s: Shapes performance patches not applied (written for %s)",
            napari.__version__,
            PATCHED_NAPARI_VERSION,
        )
        return False
    try:
        _patch_shape_list()
        _patch_mouse_bindings()
        _patch_layer()
    except Exception:  # noqa: BLE001 - never break the plugin over a speed-up
        logger.exception("Shapes performance patches failed to apply")
        return False
    _applied = True
    return True


# --------------------------------------------------------------------------- #
# 1. Bookkeeping
# --------------------------------------------------------------------------- #


def _patch_shape_list() -> None:
    from napari.layers.shapes import _shape_list as SL

    ShapeList = SL.ShapeList
    original_update_displayed = ShapeList._update_displayed

    def _build_display_gather(self, displayed: np.ndarray) -> None:
        """Gather indices and index maps of the displayed shapes.

        Built from each displayed shape's *actual* triangle/vertex range, so
        gathers and maps always agree. napari's own gather instead takes one
        slice over consecutive shapes (which includes the padding napari keeps
        in a shape's slot after it shrinks) and indexes ``triangles_z_order``
        by triangle *positions* (only right when the z order is the identity).
        """
        disp = np.nonzero(displayed)[0]
        tri_counts = np.fromiter(
            (self.shapes[i].triangles_count for i in disp),
            dtype=np.int64,
            count=len(disp),
        )
        vert_counts = np.fromiter(
            (self.shapes[i].data_displayed.shape[0] for i in disp),
            dtype=np.int64,
            count=len(disp),
        )
        self._roxas_tri_counts = dict(zip(disp.tolist(), tri_counts.tolist()))
        self._roxas_vert_counts = dict(zip(disp.tolist(), vert_counts.tolist()))

        # Triangles: displayed shapes in z order (drawn back to front)
        z_order = np.asarray(self._z_order)
        if len(z_order) != len(self.shapes):
            z_order = np.argsort(self._z_index, kind="stable")
        disp_z = z_order[displayed[z_order]]
        tri_counts_by_shape = np.zeros(len(self.shapes), dtype=np.int64)
        tri_counts_by_shape[disp] = tri_counts
        tri_counts_z = tri_counts_by_shape[disp_z]
        self._roxas_tri_gather = _concat_ranges(
            self._mesh.triangles_index[disp_z], tri_counts_z
        )
        self._roxas_tri_map = np.repeat(disp_z, tri_counts_z).astype(
            SL.IndexDtype
        )
        # Vertices: displayed shapes in index order (as napari)
        self._roxas_vert_gather = _concat_ranges(
            self._vertices_index[disp], vert_counts
        )
        self._roxas_vert_map = np.repeat(disp, vert_counts).astype(
            SL.IndexDtype
        )
        self._roxas_disp = disp

    def _update_z_order(self):
        key = (
            np.asarray(self._z_index).tobytes(),
            self._mesh.triangles_index.tobytes(),
            len(self._mesh.triangles),
        )
        if key == getattr(self, "_roxas_z_key", None):
            return  # z indices and triangle layout unchanged: same z order
        self._z_order = np.argsort(self._z_index, kind="stable")
        if len(self._z_order) == 0:
            self._mesh.triangles_z_order = np.empty(0, dtype=SL.ZOrderDtype)
        else:
            idx = self._mesh.triangles_index
            counts = np.diff(np.append(idx, len(self._mesh.triangles)))
            run_counts = counts[self._z_order]
            run_starts = idx[self._z_order]
            # Concatenation of arange(start, start + count) for each shape
            within = np.arange(run_counts.sum()) - np.repeat(
                np.cumsum(run_counts) - run_counts, run_counts
            )
            self._mesh.triangles_z_order = (
                np.repeat(run_starts, run_counts) + within
            ).astype(SL.ZOrderDtype)
        self._roxas_z_key = key
        self._roxas_z_version = getattr(self, "_roxas_z_version", 0) + 1
        self._update_displayed()

    def _displayed_mask(self) -> np.ndarray:
        if not len(self.shapes):
            return np.zeros(0, dtype=bool)
        if np.size(self.slice_key) == 0:
            # No non-displayed dims (e.g. a 2D layer): every shape is in the
            # slice. Skips ``slice_keys``, a Python loop over all shapes.
            displayed = np.ones(len(self.shapes), dtype=bool)
        else:
            slice_key = np.array([self.slice_key, self.slice_key])
            displayed = np.all(
                np.abs(self.slice_keys - slice_key) < 0.5, axis=(1, 2)
            )
        box = getattr(self, "_roxas_cull_box", None)
        if box is not None and len(self._vertices):
            lo, hi = _shape_bboxes(self)
            (y0, x0), (y1, x1) = box
            displayed &= (
                (hi[:, -2] >= y0)
                & (lo[:, -2] <= y1)
                & (hi[:, -1] >= x0)
                & (lo[:, -1] <= x1)
            )
        return displayed

    def _update_displayed(self):
        if not self._ShapeList__batch_force_call:
            # Inside a batch: only counts the call (original behaviour)
            return original_update_displayed(self)
        displayed = _displayed_mask(self)
        key = (
            displayed.tobytes(),
            getattr(self, "_roxas_z_version", 0),
            self._mesh.triangles_index.tobytes(),
            len(self._mesh.triangles),
            self._vertices_index.tobytes(),
        )
        self._displayed = displayed
        structure_changed = key != getattr(self, "_roxas_display_key", None)
        if structure_changed:
            _build_display_gather(self, displayed)
        # Contents may have changed in place: re-gather what the GPU reads
        gather = self._roxas_tri_gather
        self._mesh.displayed_triangles = self._mesh.triangles[gather]
        self._mesh.displayed_triangles_colors = self._mesh.triangles_colors[
            gather
        ]
        self.displayed_vertices = self._vertices[self._roxas_vert_gather]
        if structure_changed:
            self._mesh.displayed_triangles_to_shape_index = self._roxas_tri_map
            self.displayed_vertices_to_shape_num = self._roxas_vert_map
            self.displayed_index = self._vertices_index[self._roxas_disp]
            self._roxas_display_key = key

    original_update = ShapeList.update

    def update(self, index):
        # A shape's actual vertex/triangle count can change inside its padded
        # slot (e.g. vertex removed): the cached gathers no longer fit then.
        with self.batched_updates():
            original_update(self, index)
            shape = self.shapes[index]
            if self._roxas_tri_counts_get(index) != shape.triangles_count or (
                self._roxas_vert_counts_get(index) != shape.data_displayed.shape[0]
            ):
                self._roxas_display_key = None

    ShapeList._roxas_tri_counts_get = lambda self, i: getattr(self, "_roxas_tri_counts", {}).get(i)
    ShapeList._roxas_vert_counts_get = lambda self, i: getattr(self, "_roxas_vert_counts", {}).get(i)
    ShapeList.update = update
    ShapeList._update_z_order = SL._batch_dec(_update_z_order)
    ShapeList._update_displayed = _update_displayed


def _concat_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(start, start + count)``, vectorised."""
    total = int(counts.sum())
    offsets = np.cumsum(counts) - counts
    return np.repeat(starts - offsets, counts) + np.arange(total)


def _shape_bboxes(shape_list) -> tuple[np.ndarray, np.ndarray]:
    """Per-shape (min, max) of the displayed vertices, vectorised."""
    starts = shape_list._vertices_index[: len(shape_list.shapes)]
    vertices = shape_list._vertices
    return (
        np.minimum.reduceat(vertices, starts, axis=0),
        np.maximum.reduceat(vertices, starts, axis=0),
    )


def _patch_mouse_bindings() -> None:
    from napari.layers.shapes import _shapes_mouse_bindings as MB

    original_move = MB._move_selected_layer
    original_initiate = MB.initiate_polygon_draw

    def _move_selected_layer(layer, coordinates, vertex):
        # One display rebuild per mouse move, not one per selected shape
        with layer._data_view.batched_updates():
            return original_move(layer, coordinates, vertex)

    def initiate_polygon_draw(layer, coordinates):
        original_initiate(layer, coordinates)
        try:
            _pre_pad_shape(layer._data_view, layer.nshapes - 1)
        except Exception:  # noqa: BLE001 - padding is only a speed-up
            logger.exception("Could not pre-pad the polygon being drawn")

    MB._move_selected_layer = _move_selected_layer
    MB.initiate_polygon_draw = initiate_polygon_draw


# Slot reserved for a polygon being drawn, in vertices. Real cells have far
# fewer; a longer (lasso) polygon still works, napari then reallocates.
PRE_PAD_VERTICES = 64


def _pre_pad_shape(shape_list, index: int) -> None:
    """Reserve room in the flat arrays for the shape being drawn.

    napari keeps a shape's slot when it shrinks, but reallocates *all* the
    layer's vertex/mesh arrays (and rebuilds the z order) whenever it grows,
    i.e. on every vertex added while drawing. Growing the shape once to a
    large polygon, then shrinking it back, leaves a slot that the vertices
    added afterwards fit in.
    """
    from napari.layers.shapes._shapes_models import Polygon

    shape = shape_list.shapes[index]
    data = np.array(shape.data, copy=True)
    # Star with alternating radii: every corner is sharp, so the edge
    # triangulation gets its extra (bevel) vertices/triangles everywhere,
    # an upper bound for any real polygon with as many vertices.
    angles = np.linspace(0, 2 * np.pi, PRE_PAD_VERTICES, endpoint=False)
    radii = np.where(np.arange(PRE_PAD_VERTICES) % 2, 1.0, 0.5)
    star = np.repeat(data[:1], PRE_PAD_VERTICES, axis=0).astype(float)
    dims = list(shape.dims_displayed)
    star[:, dims[-2]] += radii * np.sin(angles)
    star[:, dims[-1]] += radii * np.cos(angles)
    with shape_list.batched_updates():
        try:
            shape_list.edit(index, star, new_type=Polygon)
        finally:
            # Always back to the shape being drawn, even if growing failed
            shape_list.edit(index, data, new_type=type(shape))


# --------------------------------------------------------------------------- #
# 2. View culling
# --------------------------------------------------------------------------- #


def _patch_layer() -> None:
    from napari.layers import Shapes

    original_extent_data = Shapes._extent_data.fget

    def _extent_data(self) -> np.ndarray:
        # Recomputed on every refresh (each mouse move while drawing); the
        # original loops over every shape in Python.
        shape_list = self._data_view
        displayed = list(self._slice_input.displayed)
        if (
            not shape_list.shapes
            or len(displayed) != self.ndim
            or len(shape_list._vertices) == 0
        ):
            return original_extent_data(self)
        # All dims displayed: the flat vertex array holds every coordinate
        # (in displayed order). Slot padding repeats a shape's own vertex.
        extent = np.empty((2, self.ndim))
        extent[0, displayed] = shape_list._vertices.min(axis=0)
        extent[1, displayed] = shape_list._vertices.max(axis=0)
        return extent

    Shapes._extent_data = property(_extent_data)

    original_update_draw = Shapes._update_draw

    def _update_draw(self, scale_factor, corner_pixels_displayed, shape_threshold):
        original_update_draw(
            self, scale_factor, corner_pixels_displayed, shape_threshold
        )
        if getattr(self, "_roxas_view_culling", False):
            _update_cull_box(self)

    Shapes._update_draw = _update_draw


def _update_cull_box(layer, force: bool = False) -> None:
    """Recompute the culled area if the view left it (or is much smaller)."""
    displayed = list(layer._slice_input.displayed)
    if len(displayed) != 2:
        _set_cull_box(layer, None)  # 3D view: no culling
        return
    view = np.asarray(layer.corner_pixels, dtype=float)[:, displayed]
    if not np.all(np.isfinite(view)) or np.any(view[1] <= view[0]):
        return
    box = getattr(layer._data_view, "_roxas_cull_box", None)
    if not force and box is not None:
        box = np.asarray(box)
        inside = np.all(view[0] >= box[0]) and np.all(view[1] <= box[1])
        view_area = np.prod(view[1] - view[0])
        box_area = np.prod(box[1] - box[0])
        if inside and view_area * 9 > box_area:
            return
    margin = (view[1] - view[0]) * CULL_MARGIN
    _set_cull_box(layer, (view[0] - margin, view[1] + margin))


def _set_cull_box(layer, box) -> None:
    shape_list = layer._data_view
    shape_list._roxas_cull_box = box
    with shape_list.batched_updates():
        shape_list._update_displayed()
    layer.refresh()


def enable_view_culling(layer) -> bool:
    """Only display/upload the shapes of ``layer`` near the camera view."""
    if not apply_shapes_patches():
        return False
    layer._roxas_view_culling = True
    _update_cull_box(layer, force=True)
    return True


def disable_view_culling(layer) -> None:
    if getattr(layer, "_roxas_view_culling", False):
        layer._roxas_view_culling = False
        _set_cull_box(layer, None)
