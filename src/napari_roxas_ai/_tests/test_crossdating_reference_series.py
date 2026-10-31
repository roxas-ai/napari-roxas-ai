"""
Tests for what "reference_series" in a sample's metadata stands for.

The entry records the reference series of an exported crossdating plot, which
is what makes it a statement about a crossdating somebody checked. Opening a
sample preselects a series by name, and persisting that would turn the entry
into "whatever the widget happened to pick", i.e. it would look crossdated
without anyone having looked at it. A sample without an exported plot keeps the
"NA" it is prepared with.
"""

import json
from types import SimpleNamespace

import pytest
from magicgui.widgets import Slider

from napari_roxas_ai._crossdating._cross_dating_plotter import (
    CrossDatingPlotterWidget,
)
from napari_roxas_ai._preparation._worker import Worker
from napari_roxas_ai._utils._metadata_keys import NO_REFERENCE_SERIES


class _Plotter(CrossDatingPlotterWidget):
    """
    The plotter reduced to what these tests touch.

    Built without its __init__, which would need a viewer, a rings layer, a
    crossdating file and a matplotlib canvas to say anything about the two
    methods under test.
    """

    def __init__(self, layer, column, metadata_path, exported=True):
        self._layer = layer
        self._crossdating_column_combo = SimpleNamespace(value=column)
        self._metadata_path = metadata_path
        self._exported = exported
        self.saved_calls = []

    @property
    def _input_layer(self):
        return self._layer

    def _sample_metadata_path(self):
        return self._metadata_path

    def save_crossdating_plot_image(self, out_dir, sample_name):
        self.saved_calls.append((out_dir, sample_name))
        return self._metadata_path if self._exported else None

    # Selecting a series redraws the plot and resets the alignment buttons,
    # neither of which exists without the real widget
    def _clear_alignment_buttons(self):
        pass

    def _sync_rings_editor_year(self):
        pass


@pytest.fixture
def sample(tmp_path):
    """A prepared sample: metadata on disk and a layer holding it."""
    path = tmp_path / "MEN.FICU_RAL16A.metadata.json"
    stored = {
        "sample_name": "MEN.FICU_RAL16A",
        "spatial_resolution": 2.2675,
        "reference_series": NO_REFERENCE_SERIES,
    }
    path.write_text(json.dumps(stored, indent=4), encoding="utf-8")

    layer = SimpleNamespace(
        name="MEN.FICU_RAL16A.rings",
        metadata=dict(stored),
    )
    return layer, path


def _stored(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_exporting_a_plot_records_its_reference_series(sample):
    layer, path = sample
    plotter = _Plotter(layer, "RAL16A", path)

    plotter._export_plot()

    assert _stored(path)["reference_series"] == "RAL16A"
    assert layer.metadata["reference_series"] == "RAL16A"


def test_applying_changes_records_reference_series(sample, monkeypatch):
    layer, path = sample
    layer.data = SimpleNamespace(shape=(10, 10))
    layer.features = None
    layer.metadata["rings_outmost_complete_year"] = 2020
    plotter = _Plotter(layer, "RAL16A", path)
    plotter._base_offset = 0
    plotter._offset_slider = Slider(value=2, min=-50, max=50)
    plotter._x_range_slider = SimpleNamespace(native=SimpleNamespace(blockSignals=lambda b: None), value=(0, 100))
    monkeypatch.setattr(
        "napari_roxas_ai._crossdating._cross_dating_plotter.update_rings_geometries",
        lambda rings_table, last_year, image_shape: (None, "new_raster", "new_cmap"),
    )
    monkeypatch.setattr(plotter, "_sync_rings_editor_year", lambda: None)
    plotter._viewer = SimpleNamespace(layers=[layer])
    annotated_calls = []
    monkeypatch.setattr(
        "napari_roxas_ai._crossdating._cross_dating_plotter.save_layers_annotated_image",
        lambda layers, stem: annotated_calls.append((layers, stem)),
    )

    plotter._apply_offset_to_layer()

    assert _stored(path)["reference_series"] == "RAL16A"
    assert layer.metadata["reference_series"] == "RAL16A"
    assert len(plotter.saved_calls) == 1
    # Applying the offset redraws the annotated image and leaves the layer to be saved
    assert annotated_calls == [([layer], layer.metadata.get("sample_stem_path"))]
    assert layer.metadata["_is_dirty"] is True


def test_a_failed_export_records_nothing(sample):
    """No plot, no statement about the crossdating."""
    layer, path = sample
    plotter = _Plotter(layer, "RAL16A", path, exported=False)

    plotter._export_plot()

    assert _stored(path)["reference_series"] == NO_REFERENCE_SERIES
    assert layer.metadata["reference_series"] == NO_REFERENCE_SERIES


def test_selecting_a_series_records_nothing(sample):
    """
    Opening a sample preselects a series and a user can try a few. None of
    that is a crossdating anybody vouched for, so none of it is written.
    """
    layer, path = sample
    plotter = _Plotter(layer, "RAL16A", path)
    plotter._y_range_slider_was_set = True

    plotter._on_new_crossdating_column()

    assert _stored(path)["reference_series"] == NO_REFERENCE_SERIES
    assert layer.metadata["reference_series"] == NO_REFERENCE_SERIES


def test_exporting_again_records_the_series_of_the_last_plot(sample):
    layer, path = sample
    plotter = _Plotter(layer, "RAL16A", path)
    plotter._export_plot()

    plotter._crossdating_column_combo.value = "RAL16B"
    plotter._export_plot()

    assert _stored(path)["reference_series"] == "RAL16B"


def test_only_the_reference_series_is_written(sample):
    """The other keys of the shared metadata file must not be touched."""
    layer, path = sample
    before = _stored(path)
    plotter = _Plotter(layer, "RAL16A", path)

    plotter._export_plot()

    after = _stored(path)
    assert after["spatial_resolution"] == before["spatial_resolution"]
    assert after["sample_name"] == before["sample_name"]
    assert set(after) == set(before)


def test_a_missing_metadata_file_is_not_created(sample, tmp_path):
    """
    The metadata file belongs to the sample, not to the plot export: a sample
    whose file is gone is a problem to report, not one to paper over with a
    file holding a single key.
    """
    layer, _ = sample
    missing = tmp_path / "gone.metadata.json"
    plotter = _Plotter(layer, "RAL16A", missing)

    plotter._export_plot()

    assert not missing.exists()


def test_no_layer_is_no_export(sample):
    _, path = sample
    plotter = _Plotter(None, "RAL16A", path)

    plotter._export_plot()

    assert _stored(path)["reference_series"] == NO_REFERENCE_SERIES
    assert plotter.saved_calls == []


# ------------------------------------------------ the key is always there


def test_a_prepared_sample_carries_the_key(tmp_path):
    """
    Every sample has the entry from the moment it is prepared, so a file
    without it means a sample prepared by a version that predates it, not a
    sample the crossdating widget has yet to touch.
    """
    path = tmp_path / "MEN.FICU_RAL16A.metadata.json"
    worker = Worker.__new__(Worker)  # _save_metadata needs no state

    worker._save_metadata({"sample_name": "MEN.FICU_RAL16A"}, str(path))

    assert _stored(path)["reference_series"] == NO_REFERENCE_SERIES


def test_preparation_keeps_a_reference_series_it_is_given(tmp_path):
    """The default must not overwrite the entry of a re-prepared sample."""
    path = tmp_path / "MEN.FICU_RAL16A.metadata.json"
    worker = Worker.__new__(Worker)

    worker._save_metadata(
        {"sample_name": "MEN.FICU_RAL16A", "reference_series": "RAL16A"},
        str(path),
    )

    assert _stored(path)["reference_series"] == "RAL16A"


def test_crossdating_plotter_auto_floats_with_margins_and_size_grip(
    make_napari_viewer, qtbot
):
    from qtpy.QtWidgets import QApplication, QSizeGrip
    from napari_roxas_ai._crossdating._cross_dating_plotter import _ProminentSizeGrip

    viewer = make_napari_viewer()
    widget = CrossDatingPlotterWidget(viewer)
    dock = viewer.window.add_dock_widget(
        widget, name="7 – Visual crossdating", area="right"
    )

    qtbot.wait(100)

    # Check floating
    assert dock.isFloating()

    # Check size grip
    assert hasattr(dock, "_roxas_size_grip")
    assert isinstance(dock._roxas_size_grip, (QSizeGrip, _ProminentSizeGrip))
    assert dock._roxas_size_grip.isVisible() == dock.isFloating()

    # Check geometry calculation on active screen
    main_window = viewer.window._qt_window
    target_screen = main_window.screen() if hasattr(main_window, "screen") else None
    if target_screen is None and hasattr(QApplication, "primaryScreen"):
        target_screen = QApplication.primaryScreen()

    if target_screen is not None:
        avail_geom = target_screen.availableGeometry()
        # Qt doesn't shrink the dock below its minimum size, which on a small
        # screen (e.g. on CI) is more than the screen minus the margins
        min_size = dock.minimumSize().expandedTo(dock.minimumSizeHint())
        expected_w = max(400, avail_geom.width() - 200, min_size.width())
        expected_h = max(300, avail_geom.height() - 200, min_size.height())
        expected_x = avail_geom.left() + 100
        expected_y = avail_geom.top() + 100
        geom = dock.geometry()
        # Verify geometry matches expected 100px margin dimensions
        assert abs(geom.width() - expected_w) <= 50
        assert abs(geom.height() - expected_h) <= 50
        assert abs(geom.x() - expected_x) <= 50
        assert abs(geom.y() - expected_y) <= 50


def test_save_crossdating_plot_image_scaling(tmp_path):
    from PIL import Image
    import pandas as pd
    import numpy as np
    from napari_roxas_ai._crossdating._cross_dating_plotter import CrossDatingPlotterWidget

    # Create dummy plot_df with overlap from 2000 to 2010 (11 years)
    years = np.arange(1995, 2016)
    layer_series = pd.Series(np.nan, index=years)
    layer_series.loc[2000:2010] = 50.0

    ref_series = pd.Series(np.nan, index=years)
    ref_series.loc[1998:2012] = 60.0

    avg_series = pd.Series(np.nan, index=years)
    avg_series.loc[1998:2012] = 55.0

    plot_df = pd.DataFrame({
        "layer_series": layer_series,
        "reference_series": ref_series,
        "average": avg_series,
    }, index=years)

    plotter = CrossDatingPlotterWidget.__new__(CrossDatingPlotterWidget)
    plotter.plot_df = plot_df
    plotter._crossdating_column_combo = SimpleNamespace(value="RAL16A")
    plotter._base_offset = 0
    plotter._offset_slider = SimpleNamespace(value=0)
    plotter._input_layer_combo = SimpleNamespace(value=None)

    out_path = plotter.save_crossdating_plot_image(tmp_path, "TEST_SAMPLE")
    assert out_path is not None
    assert out_path.exists()

    with Image.open(out_path) as img:
        w, h = img.size
        # overlap_min_ref = 2000, overlap_max_ref = 2010
        # Expected width = 50 + (20 + 2010 - 2000) * 70 = 50 + 30 * 70 = 2150
        # Expected height = 800
        expected_w = 50 + (20 + 2010 - 2000) * 70
        expected_h = 800
        assert w == expected_w
        assert h == expected_h


def test_crossdating_use_mean_ui_and_candidates(make_napari_viewer, qtbot):
    import numpy as np
    import pandas as pd

    viewer = make_napari_viewer()
    widget = CrossDatingPlotterWidget(viewer)

    # 1. Checkbox exists, unchecked by default, positioned to the left of the button
    assert hasattr(widget, "_use_mean_checkbox")
    assert widget._use_mean_checkbox.value is False
    assert widget._use_mean_checkbox.text == "Use Mean"
    assert hasattr(widget, "_auto_offset_row")
    assert widget._auto_offset_row[0] is widget._use_mean_checkbox
    assert widget._auto_offset_row[1] is widget._auto_offset_button
    assert widget._auto_offset_button.text == "Find Best Overlap"

    # 2. Test candidate computation for reference_series vs average
    years = np.arange(2000, 2020)
    sample_vals = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 25.0, 22.0, 28.0, 30.0, 35.0])
    layer_series = pd.Series(np.nan, index=years)
    layer_series.loc[2000:2009] = sample_vals

    ref_series = pd.Series(np.random.RandomState(42).randn(len(years)), index=years)
    ref_series.loc[2005:2014] = sample_vals * 2 + 5

    avg_series = pd.Series(np.random.RandomState(99).randn(len(years)), index=years)
    avg_series.loc[2010:2019] = sample_vals * 3 + 10

    plot_df = pd.DataFrame({
        "layer_series": layer_series,
        "reference_series": ref_series,
        "average": avg_series,
    }, index=years)

    widget.plot_df = plot_df

    # With use_mean=False -> best match at start_year=2005, end_year=2014
    cand_ref = widget._compute_alignment_candidates(plot_df, top_k=4, use_mean=False)
    assert len(cand_ref) >= 1
    assert cand_ref[0]["start_year"] == 2005
    assert cand_ref[0]["end_year"] == 2014
    assert pytest.approx(cand_ref[0]["corr"], 1e-4) == 1.0

    # With use_mean=True -> best match at start_year=2010, end_year=2019
    cand_avg = widget._compute_alignment_candidates(plot_df, top_k=4, use_mean=True)
    assert len(cand_avg) >= 1
    assert cand_avg[0]["start_year"] == 2010
    assert cand_avg[0]["end_year"] == 2019
    assert pytest.approx(cand_avg[0]["corr"], 1e-4) == 1.0


def test_crossdating_use_mean_interactive_toggling(make_napari_viewer, qtbot):
    import napari.layers
    import numpy as np
    import pandas as pd

    viewer = make_napari_viewer()
    widget = CrossDatingPlotterWidget(viewer)

    # Mock an input layer
    data = np.zeros((10, 10), dtype=np.uint32)
    layer = napari.layers.Labels(data, name="test_rings")
    layer.metadata["rings_outmost_complete_year"] = 2009
    viewer.add_layer(layer)

    years = np.arange(2000, 2020)
    sample_vals = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 25.0, 22.0, 28.0, 30.0, 35.0])
    layer_series = pd.Series(np.nan, index=years)
    layer_series.loc[2000:2009] = sample_vals

    ref_series = pd.Series(np.random.RandomState(42).randn(len(years)), index=years)
    ref_series.loc[2005:2014] = sample_vals * 2 + 5

    avg_series = pd.Series(np.random.RandomState(99).randn(len(years)), index=years)
    avg_series.loc[2010:2019] = sample_vals * 3 + 10

    widget.plot_df = pd.DataFrame({
        "layer_series": layer_series,
        "reference_series": ref_series,
        "average": avg_series,
    }, index=years)

    # Trigger auto align with use_mean = False (default)
    widget._auto_offset_button.native.click()
    assert widget._current_alignment_data is not None
    assert widget._current_alignment_data["start_year"] == 2005
    assert widget._current_alignment_data["end_year"] == 2014

    # Now toggle "Use Mean" -> should automatically re-align to average series
    widget._use_mean_checkbox.value = True
    assert widget._current_alignment_data is not None
    assert widget._current_alignment_data["start_year"] == 2010
    assert widget._current_alignment_data["end_year"] == 2019

    # Toggle back to unchecked -> should re-align to reference series
    widget._use_mean_checkbox.value = False
    assert widget._current_alignment_data is not None
    assert widget._current_alignment_data["start_year"] == 2005
    assert widget._current_alignment_data["end_year"] == 2014


def _offset_widget(make_napari_viewer, monkeypatch):
    """Plotter with a 10-ring ROXAS series (1991-2000) and a 1900-2099 reference."""
    import napari.layers
    import numpy as np
    import pandas as pd

    viewer = make_napari_viewer()
    data = np.zeros((10, 10), dtype=np.uint32)
    layer = napari.layers.Labels(data, name="test_rings.rings")
    layer.metadata["rings_outmost_complete_year"] = 2000
    layer.features = pd.DataFrame({"YEAR": list(range(1991, 2001)), "width": [10.0] * 10})
    viewer.add_layer(layer)

    widget = CrossDatingPlotterWidget(viewer)
    widget._crossdating_column_combo.choices = ["RefCol"]
    widget._crossdating_column_combo.value = "RefCol"

    years = np.arange(1900, 2100)
    layer_series = pd.Series(np.nan, index=years)
    layer_series.loc[1991:2000] = [10.0] * 10
    widget.plot_df = pd.DataFrame({
        "layer_series": layer_series,
        "reference_series": pd.Series(10.0, index=years),
        "average": pd.Series(10.0, index=years),
    }, index=years)

    monkeypatch.setattr(
        "napari_roxas_ai._crossdating._cross_dating_plotter.update_rings_geometries",
        lambda rings_table, last_year, image_shape: (rings_table, data, layer.colormap),
    )
    monkeypatch.setattr(widget, "_export_plot", lambda: None)
    monkeypatch.setattr(widget, "_sync_rings_editor_year", lambda: None)

    widget._plot_crossdating_data()
    return widget, layer


def test_crossdating_offset_slider_follows_year_range(make_napari_viewer, qtbot, monkeypatch):
    widget, layer = _offset_widget(make_napari_viewer, monkeypatch)

    # Initial view: series 1991-2000 with 10 years on each side
    assert tuple(widget._x_range_slider.value) == (1981, 2010)
    assert widget._offset_slider.range == (-10, 10)

    # Widening the Year Range widens the Offset slider
    widget._x_range_slider.value = (1950, 2050)
    assert widget._offset_slider.range == (-41, 50)

    widget._offset_slider.value = 40  # series at 2031-2040
    assert tuple(widget._x_range_slider.value) == (1950, 2050)

    # Narrowing the Year Range pulls the series back inside it
    widget._x_range_slider.value = (1990, 2020)
    assert widget._offset_slider.range == (-1, 20)
    assert widget._offset_slider.value == 20

    # Apply Changes -> rings_outmost_complete_year becomes 2000 + 20 = 2020
    widget._apply_changes_button.native.click()
    assert layer.metadata["rings_outmost_complete_year"] == 2020
    assert widget._base_offset == 0
    assert widget._offset_slider.value == 0


def test_crossdating_offset_slider_series_longer_than_year_range(make_napari_viewer, qtbot, monkeypatch):
    """A series longer than the Year Range can only be shifted while it covers the whole range."""
    widget, _ = _offset_widget(make_napari_viewer, monkeypatch)

    widget._x_range_slider.value = (1993, 1997)
    assert widget._offset_slider.range == (-3, 2)
    assert widget._offset_slider.value == 0


def test_crossdating_offset_slider_bounds_include_base_offset(make_napari_viewer, qtbot, monkeypatch):
    """After an alignment the slider shifts around the aligned position."""
    widget, _ = _offset_widget(make_napari_viewer, monkeypatch)

    widget._apply_alignment({"start_year": 2051, "end_year": 2060})
    assert widget._base_offset == 60
    assert widget._offset_slider.value == 0
    assert tuple(widget._x_range_slider.value) == (2041, 2070)
    assert widget._offset_slider.range == (-10, 10)
