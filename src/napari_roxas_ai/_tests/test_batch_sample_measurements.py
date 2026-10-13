"""The batch measure & update widget and its worker."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from napari_roxas_ai._measurements import _batch_sample_measurements as module
from napari_roxas_ai._settings import DEFAULT_SETTINGS, SettingsManager
from napari_roxas_ai._utils._metadata_keys import NO_REFERENCE_SERIES

IMAGE_WIDTH = 10
SPATIAL_RESOLUTION = 1.0
OUTMOST_YEAR = 2000


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """An isolated settings.json, so the tests never touch the real one."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(DEFAULT_SETTINGS, indent=4), encoding="utf-8")
    monkeypatch.setattr(SettingsManager, "_instance", None)
    monkeypatch.setattr(SettingsManager, "_settings_file", path)
    manager = SettingsManager()
    manager.set("project_directory", str(tmp_path))
    monkeypatch.setattr(module, "settings", manager)
    return manager


def _touch_sample(directory, name, cells=True, rings=True):
    directory.mkdir(parents=True, exist_ok=True)
    if cells:
        (directory / f"{name}.cells.png").touch()
    if rings:
        (directory / f"{name}.rings.tif").touch()
    return str(directory / name)


@pytest.fixture
def widget(settings, tmp_path, qtbot):
    _touch_sample(tmp_path, "A")
    _touch_sample(tmp_path / "sub", "B", cells=False)
    _touch_sample(tmp_path, "C", rings=False)
    w = module.BatchSampleMeasurementsWidget(MagicMock())
    qtbot.addWidget(w.native)
    # Children only report visible once their window is shown
    w.show()
    return w


# --- Widget -----------------------------------------------------------------


def test_long_project_directory_is_elided_to_50_characters():
    path = "C:/Users/vonarx/Desktop/" + "x" * 60 + "/ROXAS AI"
    elided = module._elide_path(path)
    assert len(elided) == 50
    assert elided.startswith("C:/Users/vonarx/")
    assert elided.endswith("/ROXAS AI")
    assert "…" in elided


def test_layout_follows_the_mockup(widget):
    assert widget._project_directory_label.value == "Project Directory:"
    texts = [
        widget._measure_cells_checkbox.text,
        widget._measure_rings_checkbox.text,
        widget._update_annotated_checkbox.text,
        widget._update_reference_checkbox.text,
        widget._handpick_files_checkbox.text,
        widget._check_settings_button.text,
    ]
    assert texts == [
        "Measure Cells",
        "Measure Rings",
        "Update Annotated image",
        "Update Cross-dating Reference Files",
        "Manually Select Files To Process",
        "Check Settings",
    ]
    assert widget._run_analysis_button.native.text() == "Measure && Update"
    assert not widget._file_selection_container.visible
    assert not widget._reverse_selection_button.visible


def test_updates_are_locked_on_while_measuring(widget):
    for checkbox in (
        widget._update_annotated_checkbox,
        widget._update_reference_checkbox,
    ):
        assert checkbox.value
        assert not checkbox.enabled

    # One measurement is enough to keep them locked
    widget._measure_cells_checkbox.value = False
    assert not widget._update_annotated_checkbox.enabled

    # Without measurements they can be chosen freely
    widget._measure_rings_checkbox.value = False
    assert widget._update_annotated_checkbox.enabled
    assert widget._update_reference_checkbox.enabled
    widget._update_annotated_checkbox.value = False
    assert widget._update_reference_checkbox.value

    # Measuring again locks them on again
    widget._measure_rings_checkbox.value = True
    assert widget._update_annotated_checkbox.value
    assert not widget._update_annotated_checkbox.enabled


def test_run_button_needs_something_to_do(widget):
    widget._measure_cells_checkbox.value = False
    widget._measure_rings_checkbox.value = False
    widget._update_annotated_checkbox.value = False
    assert widget._run_analysis_button.enabled
    widget._update_reference_checkbox.value = False
    assert not widget._run_analysis_button.enabled
    widget._update_annotated_checkbox.value = True
    assert widget._run_analysis_button.enabled


def test_samples_with_a_cells_or_rings_file_are_listed(widget, tmp_path):
    assert set(widget.samples) == {
        str(tmp_path / "A"),
        str(tmp_path / "sub" / "B"),
        str(tmp_path / "C"),
    }
    assert widget.samples[str(tmp_path / "C")]["rings"] is None


def test_manual_selection_and_reverse_selection(widget, tmp_path):
    widget._handpick_files_checkbox.value = True
    assert widget._handpick_files_checkbox.text.endswith(":")
    assert widget._file_selection_container.visible
    assert widget._reverse_selection_button.visible
    assert list(widget._file_select_widget.choices) == ["A", "C", "sub/B"]

    widget._file_select_widget.value = ["A"]
    assert list(widget._get_selected_samples()) == [str(tmp_path / "A")]

    widget._reverse_file_selection()
    assert set(widget._file_select_widget.value) == {"C", "sub/B"}
    assert set(widget._get_selected_samples()) == {
        str(tmp_path / "C"),
        str(tmp_path / "sub" / "B"),
    }

    widget._handpick_files_checkbox.value = False
    assert widget._handpick_files_checkbox.text == "Manually Select Files To Process"
    assert not widget._file_selection_container.visible
    assert len(widget._get_selected_samples()) == 3


def test_check_settings_opens_the_measurement_settings(widget):
    with patch.object(module.SettingsWidget, "open_floating") as open_floating:
        widget._open_settings()
    open_floating.assert_called_once_with(
        widget._viewer, section_name="measurements"
    )


# --- Worker -----------------------------------------------------------------


def _worker(**kwargs):
    options = dict(
        measure_cells=False,
        measure_rings=False,
        update_annotated_images=False,
        update_reference_plots=False,
    )
    options.update(kwargs)
    return module.Worker({}, {}, **options)


def _rings(widths, reference_series=NO_REFERENCE_SERIES):
    """Rings as read from a file, with the given ring widths (µm)."""
    widths = np.asarray(widths, dtype=float)
    years = np.arange(OUTMOST_YEAR - len(widths) + 1, OUTMOST_YEAR + 1)
    features = pd.DataFrame(
        {
            "YEAR": years,
            "cells_above": np.cumsum(widths) * IMAGE_WIDTH * SPATIAL_RESOLUTION,
            "RBXY": [[[0, 0], [0, IMAGE_WIDTH - 1]]] * len(widths),
        }
    )
    metadata = {
        "sample_name": "S1",
        "spatial_resolution": SPATIAL_RESOLUTION,
        "rings_outmost_complete_year": OUTMOST_YEAR,
        "reference_series": reference_series,
    }
    data = np.zeros((10, IMAGE_WIDTH), dtype=np.uint8)
    return data, {"features": features, "metadata": metadata}


@pytest.fixture
def sample_widths():
    rng = np.random.default_rng(0)
    return rng.uniform(50, 150, 20)


@pytest.fixture
def project(tmp_path, settings, sample_widths):
    """A project with a crossdating file in its root and a sample below it."""
    rng = np.random.default_rng(1)
    years = np.arange(1950, 2011)
    noise = pd.Series(rng.uniform(50, 150, len(years)), index=years)
    # "MATCH" holds the sample's widths at its dating, "OTHER" is unrelated
    match = noise.copy()
    match.loc[OUTMOST_YEAR - len(sample_widths) + 1 : OUTMOST_YEAR] = sample_widths
    other = pd.Series(rng.uniform(50, 150, len(years)), index=years)
    table = pd.DataFrame({"OTHER": other, "MATCH": match}, index=years)
    table.index.name = "YEAR"
    table.to_csv(tmp_path / "ref.crossdating.txt", sep="\t")

    sample_dir = tmp_path / "trees"
    sample_dir.mkdir()
    return str(sample_dir / "S1")


def test_reference_plot_uses_the_stored_series(project, sample_widths):
    with patch.object(module, "save_crossdating_plot") as save:
        _worker()._update_reference_plot(
            project, _rings(sample_widths, reference_series="OTHER")
        )
    plot_df, reference_name, image_id, out_path = save.call_args.args
    assert reference_name == "OTHER"
    assert image_id == "S1"
    assert str(out_path).endswith("trees/S1_ReferenceSeries.jpg")
    # Dated by the outmost complete ring year of the metadata
    assert plot_df["layer_series"].dropna().index.max() == OUTMOST_YEAR


def test_reference_plot_without_stored_series_uses_the_best_match(
    project, sample_widths
):
    with patch.object(module, "save_crossdating_plot") as save:
        _worker()._update_reference_plot(project, _rings(sample_widths))
    assert save.call_args.args[1] == "MATCH"


def test_reference_plot_with_unknown_stored_series_uses_the_best_match(
    project, sample_widths
):
    with patch.object(module, "save_crossdating_plot") as save:
        _worker()._update_reference_plot(
            project, _rings(sample_widths, reference_series="GONE")
        )
    assert save.call_args.args[1] == "MATCH"


def test_reference_plot_is_skipped_without_crossdating_file(
    tmp_path, settings, sample_widths
):
    with patch.object(module, "save_crossdating_plot") as save, patch.object(
        module, "find_crossdating_files", return_value=[]
    ):
        _worker()._update_reference_plot(
            str(tmp_path / "S1"), _rings(sample_widths)
        )
    save.assert_not_called()


def test_reference_plot_is_written(project, sample_widths, tmp_path):
    _worker()._update_reference_plot(project, _rings(sample_widths))
    assert (tmp_path / "trees" / "S1_ReferenceSeries.jpg").exists()


def test_updates_without_measurements_read_the_rings_file(tmp_path, settings):
    rings = _rings(np.full(10, 100.0))
    stem = str(tmp_path / "S1")
    worker = module.Worker(
        {},
        {stem: {"cells": None, "rings": f"{stem}.rings.tif"}},
        measure_cells=False,
        measure_rings=False,
        update_annotated_images=True,
        update_reference_plots=True,
    )
    with patch.object(
        module, "read_rings_file", return_value=(*rings, "labels")
    ) as read, patch.object(
        module, "save_sample_annotated_image"
    ) as annotate, patch.object(
        worker, "_update_reference_plot"
    ) as reference, patch.object(
        module, "get_software_version", return_value="test"
    ), patch.object(
        module, "get_measurement_operator", return_value="test"
    ):
        worker.run()

    read.assert_called_once()
    annotate.assert_called_once()
    assert annotate.call_args.args[0] == stem
    reference.assert_called_once()


def test_measuring_rings_does_not_redraw_the_annotated_image_twice(
    tmp_path, settings
):
    """Writing the rings file already updates the annotated image."""
    stem = str(tmp_path / "S1")
    worker = module.Worker(
        {},
        {stem: {"cells": None, "rings": f"{stem}.rings.tif"}},
        measure_cells=False,
        measure_rings=True,
        update_annotated_images=True,
        update_reference_plots=False,
    )
    rings = _rings(np.full(10, 100.0))
    with patch.object(
        worker, "_measure", return_value=(rings, True)
    ), patch.object(module, "save_sample_annotated_image") as annotate:
        worker._sw_version = worker._meas_by = "test"
        worker.run()
    annotate.assert_not_called()
