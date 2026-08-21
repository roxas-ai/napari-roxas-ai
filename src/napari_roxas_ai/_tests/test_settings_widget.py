"""
Tests for the settings widget.

The widget replaces hand-editing settings.json, so the property that matters
most is that it is lossless: every value must survive being rendered into an
editor and read back, with its type and its precision intact. A widget that
silently turns 2.2675 into 2.27 or 3.0 into 3 would be worse than the text
editor it replaces.
"""

import json
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from napari_roxas_ai._settings._settings_manager import (
    DEFAULT_SETTINGS,
    SettingsManager,
)
from napari_roxas_ai._settings._settings_widget import (
    SettingsWidget,
    _decimals_for,
    _make_editor,
)


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """An isolated settings.json, so the tests never touch the real one."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(DEFAULT_SETTINGS, indent=4), encoding="utf-8")
    monkeypatch.setattr(SettingsManager, "_instance", None)
    monkeypatch.setattr(SettingsManager, "_settings_file", path)
    return path


@pytest.fixture
def widget(settings_file, qtbot):
    """The settings widget, built against the isolated settings file."""
    settings_widget = SettingsWidget(MagicMock())
    qtbot.addWidget(settings_widget.native)
    return settings_widget


def _editor_for(widget, *path):
    """
    The editor behind a dotted settings path.

    The collectors are closures, so the widgets are reached through the Qt tree
    instead: find the row label matching the last key inside the group hierarchy.
    """
    from qtpy.QtWidgets import QFormLayout, QLabel

    def find_row(root, key):
        for form in root.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QFormLayout.LabelRole)
                field_item = form.itemAt(row, QFormLayout.FieldRole)
                if label_item is None or field_item is None:
                    continue
                label = label_item.widget()
                if isinstance(label, QLabel) and label.text() == f"{key}:":
                    return field_item.widget()
        return None

    editor = find_row(widget.native, path[-1])
    assert editor is not None, f"no editor for {'.'.join(path)}"
    return editor


# ---------------------------------------------------------------- losslessness


def test_apply_without_edits_leaves_the_file_untouched(widget, settings_file):
    """
    The strongest guarantee: opening the widget and applying must be a no-op.

    If any editor rounded, retyped or dropped a value, this is where it shows.
    """
    before = settings_file.read_text(encoding="utf-8")

    widget._apply()

    assert settings_file.read_text(encoding="utf-8") == before


def test_collect_reproduces_the_settings(widget):
    assert widget._form.collect() == json.loads(
        json.dumps(DEFAULT_SETTINGS)
    )


def test_value_types_are_preserved(widget):
    collected = widget._form.collect()

    assert isinstance(
        collected["measurements"]["cluster_dbl_cwt_threshold"], float
    )
    assert isinstance(collected["rasterization"]["uncomplete_ring_value"], int)
    assert isinstance(collected["processing"]["try_to_use_gpu"], bool)
    assert collected["project_directory"] is None
    extensions = collected["file_extensions"]["scan_file_extension"]
    assert isinstance(extensions, list)


def test_float_precision_is_not_lost(widget):
    """A QDoubleSpinBox at its default 2 decimals would give back 2.27."""
    collected = widget._form.collect()
    resolution = next(
        field
        for field in collected["samples_metadata"]["fields"]
        if field["id"] == "spatial_resolution"
    )
    assert resolution["default"] == 2.2675
    assert resolution["min"] == 0.001


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3.0, 4), (0.75, 4), (2.2675, 4), (0.001, 4), (1.23456789, 8)],
)
def test_decimals_never_truncate_the_stored_value(value, expected):
    assert _decimals_for(value) == expected
    editor, getter = _make_editor(value)
    assert getter() == value


# ------------------------------------------------------------------ the editors


def test_every_leaf_has_an_editor(widget):
    """No value may be missing from the form, or applying would drop it."""

    def leaves(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaves(value, f"{path}.{key}" if path else key)
        elif (
            isinstance(node, list)
            and node
            and all(isinstance(item, dict) for item in node)
        ):
            for index, item in enumerate(node):
                yield from leaves(item, f"{path}[{index}]")
        else:
            yield path

    collected = widget._form.collect()
    assert sorted(leaves(collected)) == sorted(leaves(DEFAULT_SETTINGS))


def test_editing_a_value_applies_to_memory_and_to_the_file(
    widget, settings_file
):
    _editor_for(widget, "processing", "try_to_use_gpu").setChecked(True)
    _editor_for(widget, "tables", "separator").setText(",")

    widget._apply()

    manager = SettingsManager()
    assert manager.get("processing.try_to_use_gpu") is True
    assert manager.get("tables.separator") == ","

    on_disk = json.loads(settings_file.read_text(encoding="utf-8"))
    assert on_disk["processing"]["try_to_use_gpu"] is True
    assert on_disk["tables"]["separator"] == ","


def test_project_directory_round_trips_through_an_empty_field(
    widget, settings_file
):
    """An empty field means unset, i.e. null, not an empty string."""
    editor = _editor_for(widget, "project_directory")
    editor.setText("C:/Projects/Test")
    widget._apply()
    assert SettingsManager().get("project_directory") == "C:/Projects/Test"

    editor.setText("   ")
    widget._apply()
    assert SettingsManager().get("project_directory") is None
    assert (
        json.loads(settings_file.read_text(encoding="utf-8"))[
            "project_directory"
        ]
        is None
    )


def test_string_list_is_edited_one_entry_per_line(widget):
    editor = _editor_for(widget, "rasterization", "rings_color_sequence")
    editor.setPlainText("red\n  blue  \n\n\ngreen\n")

    widget._apply()

    assert SettingsManager().get("rasterization.rings_color_sequence") == [
        "red",
        "blue",
        "green",
    ]


def test_a_metadata_field_can_be_edited(widget):
    """The field ids become the keys of a sample's metadata file."""
    _editor_for(widget, "label").setText("Sample Name")

    widget._apply()

    fields = SettingsManager().get("samples_metadata.fields")
    assert fields[0]["label"] == "Sample Name"
    assert fields[0]["id"] == "sample_name"  # untouched


# --------------------------------------------------------------- reload / reset


def test_reload_discards_unapplied_edits_and_reads_the_file(
    widget, settings_file
):
    _editor_for(widget, "tables", "separator").setText("#")

    external = deepcopy(DEFAULT_SETTINGS)
    external["tables"]["separator"] = "|"
    settings_file.write_text(json.dumps(external, indent=4), encoding="utf-8")

    widget._reload()

    assert widget._form.collect()["tables"]["separator"] == "|"
    assert SettingsManager().get("tables.separator") == "|"


def test_reset_restores_the_defaults_after_confirmation(widget):
    _editor_for(widget, "tables", "separator").setText("#")
    widget._apply()

    with patch(
        "napari_roxas_ai._settings._settings_widget.QMessageBox"
    ) as message_box:
        message_box.question.return_value = message_box.Yes
        widget._reset()

    assert widget._form.collect() == json.loads(json.dumps(DEFAULT_SETTINGS))


def test_reset_is_abandoned_when_not_confirmed(widget):
    _editor_for(widget, "tables", "separator").setText("#")
    widget._apply()

    with patch(
        "napari_roxas_ai._settings._settings_widget.QMessageBox"
    ) as message_box:
        message_box.question.return_value = message_box.No
        widget._reset()

    assert SettingsManager().get("tables.separator") == "#"


# ------------------------------------------------------------------- the layout


def test_the_buttons_stay_below_the_form(widget):
    """Rebuilding the form must not push the buttons up."""
    layout = widget.native.layout()

    def order():
        return [
            type(layout.itemAt(index).widget()).__name__
            for index in range(layout.count())
        ]

    before = order()
    widget._build_form()

    assert before == order()
    assert layout.indexOf(widget._form.native) < layout.indexOf(
        widget._apply_button.native
    )


# ----------------------------------------------------- refreshing open widgets


def test_apply_refreshes_open_widgets_that_ask_for_it(widget):
    """
    Widgets opt into a refresh by defining refresh_from_settings().

    Everything that reads a setting when it needs it follows the singleton on
    its own; only values a widget copied while being built need the nudge.
    The public dock_widgets mapping yields the inner widget of each dock, so
    there is nothing to unwrap here.
    """
    refreshing = MagicMock(spec=["refresh_from_settings"])
    plain = MagicMock(spec=[])  # no refresh_from_settings

    widget._viewer.window.dock_widgets = {
        "refreshing": refreshing,
        "plain": plain,
    }

    widget._apply()

    refreshing.refresh_from_settings.assert_called_once_with()


def test_apply_does_not_refresh_the_settings_widget_itself(widget):
    """
    Its own refresh_from_settings() rebuilds the form, which would throw away
    the editors mid-apply.
    """
    widget._viewer.window.dock_widgets = {"ZZ - Settings": widget}
    form_before = widget._form

    widget._apply()

    assert widget._form is form_before


def test_a_failing_refresh_does_not_break_apply(widget, settings_file):
    broken = MagicMock(spec=["refresh_from_settings"])
    broken.refresh_from_settings.side_effect = RuntimeError("boom")
    widget._viewer.window.dock_widgets = {"broken": broken}

    _editor_for(widget, "tables", "separator").setText(";;")
    widget._apply()

    assert SettingsManager().get("tables.separator") == ";;"
def _touch_every_editor(widget):
    """
    Write every editor's own current value back into it through the Qt API.

    Applying without touching anything would also pass if the editors never
    represented the values at all. This forces every value through the widget
    that owns it, which is what catches a spinbox clamping its range or turning
    a bool into 1.
    """
    from qtpy.QtWidgets import (
        QCheckBox,
        QDoubleSpinBox,
        QLineEdit,
        QPlainTextEdit,
        QSpinBox,
    )

    for editor in widget.native.findChildren(QCheckBox):
        editor.setChecked(editor.isChecked())
    for editor in widget.native.findChildren(QSpinBox):
        editor.setValue(editor.value())
    for editor in widget.native.findChildren(QDoubleSpinBox):
        editor.setValue(editor.value())
    for editor in widget.native.findChildren(QLineEdit):
        editor.setText(editor.text())
    for editor in widget.native.findChildren(QPlainTextEdit):
        editor.setPlainText(editor.toPlainText())


def _assert_same_types(expected, actual, path=""):
    """Equal is not enough: True == 1 and 3.0 == 3 must still fail here."""
    assert type(expected) is type(
        actual
    ), f"{path}: {expected!r} -> {actual!r}"

    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            _assert_same_types(
                expected[key], actual[key], f"{path}.{key}" if path else key
            )
    elif isinstance(expected, list):
        assert len(expected) == len(actual), path
        for index, item in enumerate(expected):
            _assert_same_types(item, actual[index], f"{path}[{index}]")
    else:
        assert expected == actual, path


def test_every_value_survives_its_own_editor(widget, settings_file):
    """Round trip through the widgets, not around them."""
    _touch_every_editor(widget)

    widget._apply()

    _assert_same_types(
        json.loads(json.dumps(DEFAULT_SETTINGS)),
        json.loads(settings_file.read_text(encoding="utf-8")),
    )


def test_numeric_editors_can_hold_their_value(widget):
    """A spinbox whose range or precision excludes its own value is a bug."""
    from qtpy.QtWidgets import QDoubleSpinBox, QSpinBox

    spin_boxes = widget.native.findChildren(QSpinBox)
    float_spin_boxes = widget.native.findChildren(QDoubleSpinBox)
    assert spin_boxes and float_spin_boxes  # the form really has both

    for editor in spin_boxes + float_spin_boxes:
        assert editor.minimum() <= editor.value() <= editor.maximum()

    for editor in float_spin_boxes:
        value = editor.value()
        assert float(f"%.{editor.decimals()}f" % value) == value


def test_unknown_shapes_are_passed_through(settings_file, monkeypatch, qtbot):
    """
    A user file may hold shapes the defaults never do.

    _merge_defaults keeps keys that only exist in the user's file, so the widget
    must not drop or retype them just because it has no editor for them.
    """
    exotic = json.loads(settings_file.read_text(encoding="utf-8"))
    exotic["my_own_block"] = {
        "numbers": [1, 2, 3],
        "nested": [[1], [2]],
        "mixed": ["a", 1, None],
        "deep": {"a": {"b": {"c": 1}}},
    }
    settings_file.write_text(json.dumps(exotic, indent=4), encoding="utf-8")
    monkeypatch.setattr(SettingsManager, "_instance", None)

    widget = SettingsWidget(MagicMock())
    qtbot.addWidget(widget.native)
    widget._apply()

    _assert_same_types(
        exotic["my_own_block"],
        json.loads(settings_file.read_text(encoding="utf-8"))["my_own_block"],
    )


def test_a_metadata_field_id_cannot_be_edited(widget):
    """
    Renaming an id would add a field rather than rename it.

    upgrade_settings() re-inserts the default field under its original id and
    keeps the renamed one as a user addition, and the ids are the keys of a
    sample's metadata file, so they are shown read-only.
    """
    editor = _editor_for(widget, "id")
    assert editor.isReadOnly()


def test_an_emptied_file_extension_is_refused(widget, settings_file):
    """The extensions are read as [0], so an empty list breaks load and save."""
    before = settings_file.read_text(encoding="utf-8")
    _editor_for(widget, "scan_file_extension").setPlainText("")

    with patch(
        "napari_roxas_ai._settings._settings_widget.QMessageBox"
    ) as message_box:
        widget._apply()
        message_box.warning.assert_called_once()

    assert settings_file.read_text(encoding="utf-8") == before


def test_reload_of_a_broken_file_keeps_the_settings_in_use(
    widget, settings_file
):
    """_load_settings() would fall back to the defaults, losing good values."""
    _editor_for(widget, "tables", "separator").setText("#")
    widget._apply()

    settings_file.write_text("{ broken", encoding="utf-8")

    with patch(
        "napari_roxas_ai._settings._settings_widget.QMessageBox"
    ) as message_box:
        widget._reload()
        message_box.warning.assert_called_once()

    assert SettingsManager().get("tables.separator") == "#"


def test_reload_of_a_deleted_file_does_not_overwrite_it(widget, settings_file):
    widget._apply()
    settings_file.unlink()

    with patch("napari_roxas_ai._settings._settings_widget.QMessageBox"):
        widget._reload()

    assert not settings_file.exists()
