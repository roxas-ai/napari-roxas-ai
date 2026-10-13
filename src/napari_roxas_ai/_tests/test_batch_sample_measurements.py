"""The batch measure & update widget and its worker."""

import json
from pathlib import Path
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


def test_long_project_directory_is_elided_to_35_characters():
    path = "C:/Users/vonarx/Desktop/" + "x" * 60 + "/ROXAS AI"
    elided = module._elide_path(path)
    assert len(elided) == 35
    assert elided.startswith("C:/Users/vonarx/")
    assert elided.endswith("/ROXAS AI")
    assert "…" in elided


def test_layout_follows_the_mockup(widget):
    assert widget._project_directory_label.value == "Project Directory:"
    # File selection first, then what to do, then the buttons
    rows = list(widget)
    order = [
        widget._project_directory_row,
        widget._handpick_files_checkbox,
        widget._file_selection_container,
        widget._reverse_selection_button,
        widget._measure_cells_checkbox,
        widget._measure_rings_checkbox,
        widget._update_annotated_checkbox,
        widget._update_reference_checkbox,
        widget._check_settings_button,
        widget._run_analysis_button,
    ]
    assert [rows.index(w) for w in order] == sorted(rows.index(w) for w in order)

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
        "Update Crossdating Reference Files",
        "Manually Select Files To Process:",
        "Check Settings",
    ]
    assert widget._run_analysis_button.native.text() == "Measure && Update"
    assert widget._file_selection_container.visible
    assert widget._reverse_selection_button.visible


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
    assert widget._handpick_files_checkbox.value is True
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


def test_file_list_is_refilled_not_replaced(widget, settings, tmp_path):
    """
    A replaced list would be left to the garbage collector, which may destroy
    it in the worker thread and crash napari (Qt timers of the main thread).
    """
    widget._handpick_files_checkbox.value = True
    file_list = widget._file_select_widget

    _touch_sample(tmp_path, "D")
    widget._refresh_file_list()
    widget._handpick_files_checkbox.value = False
    widget._handpick_files_checkbox.value = True

    assert widget._file_select_widget is file_list
    assert "D" in file_list.choices
    assert list(widget._file_selection_container) == [file_list]


def test_file_list_choices_retained_on_reset_choices_or_docking(
    widget, make_napari_viewer
):
    """napari calls reset_choices() when adding dock widgets."""
    assert len(widget._file_select_widget.choices) > 0
    widget._file_select_widget.reset_choices()
    assert len(widget._file_select_widget.choices) > 0


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


def _rings(widths, sample_name="S1", reference_series=NO_REFERENCE_SERIES):
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
        "sample_name": sample_name,
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
    match = noise.copy()
    match.loc[OUTMOST_YEAR - len(sample_widths) + 1 : OUTMOST_YEAR] = sample_widths
    other = pd.Series(rng.uniform(50, 150, len(years)), index=years)
    table = pd.DataFrame({"OTHER": other, "S1": match}, index=years)
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
    assert Path(out_path).parts[-2:] == ("trees", "S1_ReferenceSeries.jpg")
    # Dated by the outmost complete ring year of the metadata
    assert plot_df["layer_series"].dropna().index.max() == OUTMOST_YEAR


def test_reference_plot_without_stored_series_uses_the_matching_name(
    project, sample_widths
):
    with patch.object(module, "save_crossdating_plot") as save, patch.object(
        module, "update_metadata_file"
    ) as update_meta:
        task = _worker()._update_reference_plot(project, _rings(sample_widths))
    assert save.call_args.args[1] == "S1"
    assert task == "Reference plot (S1, best match)"
    update_meta.assert_called_once()
    assert update_meta.call_args.args[1]["reference_series"] == "S1"


def test_reference_plot_with_unknown_stored_series_uses_the_matching_name(
    project, sample_widths
):
    with patch.object(module, "save_crossdating_plot") as save, patch.object(
        module, "update_metadata_file"
    ) as update_meta:
        _worker()._update_reference_plot(
            project, _rings(sample_widths, reference_series="GONE")
        )
    assert save.call_args.args[1] == "S1"
    update_meta.assert_called_once()
    assert update_meta.call_args.args[1]["reference_series"] == "S1"


def test_reference_plot_without_matching_series_is_skipped(
    project, sample_widths
):
    # Sample named "UNMATCHED" does not match columns "OTHER" or "S1"
    sample_dir = Path(project).parent
    unmatched_stem = str(sample_dir / "UNMATCHED")
    rings = _rings(sample_widths, sample_name="UNMATCHED")
    # Missing reference_series in metadata
    del rings[1]["metadata"]["reference_series"]
    with patch.object(module, "save_crossdating_plot") as save, patch.object(
        module, "update_metadata_file"
    ) as update_meta:
        task = _worker()._update_reference_plot(unmatched_stem, rings)
    save.assert_not_called()
    assert task == "Reference plot skipped (no matching reference series)"
    update_meta.assert_called_once()
    assert update_meta.call_args.args[1]["reference_series"] == NO_REFERENCE_SERIES


def test_reference_plot_is_skipped_without_crossdating_file(
    tmp_path, settings, sample_widths
):
    rings = _rings(sample_widths)
    del rings[1]["metadata"]["reference_series"]
    with patch.object(module, "save_crossdating_plot") as save, patch.object(
        module, "find_crossdating_files", return_value=[]
    ), patch.object(module, "update_metadata_file") as update_meta:
        task = _worker()._update_reference_plot(
            str(tmp_path / "S1"), rings
        )
    save.assert_not_called()
    assert task == "Reference plot skipped (no crossdating file)"
    update_meta.assert_called_once()
    assert update_meta.call_args.args[1]["reference_series"] == NO_REFERENCE_SERIES


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
        worker, "_measure", return_value=(rings, True, "Rings")
    ), patch.object(module, "save_sample_annotated_image") as annotate:
        worker.run()
    annotate.assert_not_called()


def test_worker_prints_info_for_each_processed_file(tmp_path, settings, capsys):
    """Each sample is reported in the terminal, as in batch detection."""
    stems = [str(tmp_path / "S1"), str(tmp_path / "S2")]
    worker = module.Worker(
        {},
        {stem: {"cells": None, "rings": f"{stem}.rings.tif"} for stem in stems},
        measure_cells=False,
        measure_rings=True,
        update_annotated_images=True,
        update_reference_plots=True,
    )
    rings = _rings(np.full(10, 100.0))
    with patch.object(
        worker,
        "_measure",
        side_effect=[(rings, True, "Rings"), RuntimeError("broken")],
    ), patch.object(
        worker, "_update_reference_plot", return_value="Reference plot (A)"
    ):
        worker.run()

    out = capsys.readouterr().out
    assert (
        "[Batch Measurements] [1/2] S1: Rings, Annotated image, Reference plot (A)"
        in out
    )
    assert "[Batch Measurements] [2/2] Error processing S2: broken" in out
    assert "[Batch Measurements] Completed 1 of 2 file(s) successfully." in out


def test_reference_plot_far_from_its_reference_is_saved(tmp_path):
    """
    A sample without overlap is plotted over the whole gap to its reference,
    which used to make the JPEG wider than its 65,535 pixel limit.
    """
    from napari_roxas_ai._crossdating._cross_dating_plotter import (
        MAX_REFERENCE_PLOT_WIDTH_PX,
        build_crossdating_plot_df,
        save_crossdating_plot,
    )

    reference_years = np.arange(400, 500)
    crossdating = pd.DataFrame(
        {"REF": np.linspace(50, 150, len(reference_years))},
        index=reference_years,
    )
    crossdating["average"] = crossdating["REF"]
    widths = pd.Series(np.full(20, 100.0), index=np.arange(9980, 10000))
    plot_df = build_crossdating_plot_df(crossdating, "REF", widths)

    out_path = save_crossdating_plot(
        plot_df, "REF", "S1", tmp_path / "S1_ReferenceSeries.jpg"
    )

    from PIL import Image

    with Image.open(out_path) as image:
        assert image.width <= MAX_REFERENCE_PLOT_WIDTH_PX
