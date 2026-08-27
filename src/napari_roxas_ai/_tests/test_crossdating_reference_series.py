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

    def _update_crossdating_plot(self):
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
