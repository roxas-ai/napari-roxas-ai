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
from qtpy.QtWidgets import QApplication

from napari_roxas_ai._settings import _settings_widget
from napari_roxas_ai._settings._settings_manager import (
    DEFAULT_SETTINGS,
    SettingsManager,
)
from napari_roxas_ai._settings._settings_widget import (
    INFO_ICON,
    SETTING_HINTS,
    SettingsWidget,
    _InfoIcon,
    _decimals_for,
    _list_editor_style,
    _make_editor,
    _row_label,
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

    def label_texts(label):
        """A row label is a plain QLabel, or a widget holding one plus an icon."""
        if isinstance(label, QLabel):
            return [label.text()]
        return [inner.text() for inner in label.findChildren(QLabel)]

    def find_row(root, key):
        for form in root.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QFormLayout.LabelRole)
                field_item = form.itemAt(row, QFormLayout.FieldRole)
                if label_item is None or field_item is None:
                    continue
                if f"{key}:" in label_texts(label_item.widget()):
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


# ------------------------------------------------------ the explanation hints


def _row_label_widget(widget, key):
    """The label widget of the row belonging to a settings key."""
    from qtpy.QtWidgets import QFormLayout, QLabel

    for form in widget.native.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.LabelRole)
            if item is None:
                continue
            label = item.widget()
            texts = (
                [label.text()]
                if isinstance(label, QLabel)
                else [inner.text() for inner in label.findChildren(QLabel)]
            )
            if f"{key}:" in texts:
                return label
    return None


def test_every_hint_belongs_to_a_setting_that_exists():
    """
    The hints repeat the comments in DEFAULT_SETTINGS, so they can fall behind
    a renamed or removed setting. A hint whose path is gone shows up nowhere in
    the widget, which is exactly why it would go unnoticed.
    """

    def paths(mapping, prefix=""):
        for key, value in mapping.items():
            yield f"{prefix}{key}"
            if isinstance(value, dict):
                yield from paths(value, f"{prefix}{key}.")

    unknown = set(SETTING_HINTS) - set(paths(DEFAULT_SETTINGS))
    assert not unknown


def test_an_explained_setting_gets_an_icon_with_its_explanation(widget):
    from qtpy.QtWidgets import QLabel

    label = _row_label_widget(widget, "try_to_use_gpu")
    icons = [
        inner
        for inner in label.findChildren(QLabel)
        if inner.text() == INFO_ICON
    ]

    assert len(icons) == 1
    assert icons[0].toolTip() == "Try to use GPU if available"
    # Hovering the name works as well as hovering the icon
    assert label.toolTip() == "Try to use GPU if available"


def _click(icon):
    """
    Press and release on an icon, with the event loop running in between.

    A real click arrives as two events some hundred milliseconds apart, and the
    difference matters here: anything scheduled on the press would be undone by
    the release.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest

    QTest.mousePress(icon, Qt.LeftButton)
    QApplication.instance().processEvents()
    QTest.mouseRelease(icon, Qt.LeftButton)


def _icon_for(widget, hint):
    """The info icon carrying the given explanation."""
    return next(
        candidate
        for candidate in widget.native.findChildren(_InfoIcon)
        if candidate.toolTip() == hint
    )


def test_clicking_the_icon_asks_for_the_explanation_after_the_click(widget):
    """
    A tooltip needs the pointer held still on a small target, which a trackpad
    makes fiddly; a click has to work as well.

    Qt takes the visible tooltip down on every mouse press and release, so
    showing one from inside either handler makes it appear and vanish again --
    it has to go up once the click is over. That is what is checked here rather
    than the tooltip being on screen: whether a tooltip renders at all depends
    on the platform plugin, and the CI runner is not the machine the widget
    runs on.
    """
    icon = _icon_for(widget, "Try to use GPU if available")

    with patch.object(_settings_widget.QToolTip, "showText") as show_text:
        _click(icon)

        # Not while the click is being delivered, or Qt hides it right away
        assert not show_text.called

        QApplication.instance().processEvents()

    assert show_text.called
    _, text, owner = show_text.call_args[0]
    assert text == "Try to use GPU if available"
    assert owner is icon


def test_clicking_the_icon_again_asks_again(widget):
    icon = _icon_for(widget, "Try to use GPU if available")

    with patch.object(_settings_widget.QToolTip, "showText") as show_text:
        _click(icon)
        QApplication.instance().processEvents()
        _click(icon)
        QApplication.instance().processEvents()

    assert show_text.call_count == 2


def test_the_explanation_is_shown_on_the_icon(widget):
    """At the icon, not wherever the pointer happens to be."""
    icon = _icon_for(widget, "Try to use GPU if available")

    with patch.object(_settings_widget.QToolTip, "showText") as show_text:
        icon._show_explanation()

    position, _, _ = show_text.call_args[0]
    assert position == icon.mapToGlobal(icon.rect().center())


def test_the_icon_shows_that_it_can_be_clicked(widget):
    from qtpy.QtCore import Qt

    icon = widget.native.findChild(_InfoIcon)

    assert icon.cursor().shape() == Qt.PointingHandCursor


def test_a_setting_without_an_explanation_gets_no_icon(widget):
    """The icon has to mean there is something to read."""
    from qtpy.QtWidgets import QLabel

    label = _row_label_widget(widget, "separator")

    assert isinstance(label, QLabel)
    assert label.text() == "separator:"
    assert not label.toolTip()
    assert INFO_ICON not in label.text()


def test_the_explanations_reach_the_nested_sections(widget):
    """The hints are keyed by dotted path, so the nesting has to be tracked."""
    label = _row_label_widget(widget, "cluster_dbl_cwt_threshold")

    assert label.toolTip() == "Default cluster DBL/CWT threshold in \u00b5m"


def test_row_label_is_a_plain_label_without_a_hint():
    from qtpy.QtWidgets import QLabel

    assert isinstance(_row_label("quality", ""), QLabel)
    assert not isinstance(_row_label("quality", "an explanation"), QLabel)


def test_the_icons_do_not_change_what_the_form_holds(widget):
    """A composite row label must not disturb the editors it labels."""
    assert widget._form.collect() == json.loads(json.dumps(DEFAULT_SETTINGS))


# --------------------------------------------------------------- the filter


def _visible_settings(widget):
    """The paths of the settings the form is currently showing."""
    return {
        entry.path
        for entry in _leaf_entries(widget._form._entries)
        if all(w.isVisible() for w in entry.widgets)
    }


def _leaf_entries(entries):
    for entry in entries:
        if entry.children:
            yield from _leaf_entries(entry.children)
        else:
            yield entry


def _filter(widget, text):
    """Type into the filter field, as a user does."""
    from qtpy.QtWidgets import QApplication

    widget.native.resize(430, 900)
    widget.native.show()
    widget._filter_field.native.setText(text)
    for _ in range(3):  # the sections settle over a couple of layout passes
        QApplication.instance().processEvents()


def test_the_filter_field_is_the_topmost_widget(widget):
    layout = widget.native.layout()

    assert layout.indexOf(widget._filter_field.native) == 0


def test_the_filter_matches_part_of_a_name_ignoring_case(widget):
    """"gpu" has to find try_to_use_gpu."""
    for text in ("gpu", "GPU", "Gpu"):
        _filter(widget, text)
        assert _visible_settings(widget) == {"processing.try_to_use_gpu"}


def test_the_filter_keeps_every_setting_that_matches(widget):
    _filter(widget, "cwt")

    visible = _visible_settings(widget)
    assert all("cwt" in path for path in visible)
    assert "measurements.cluster_dbl_cwt_threshold" in visible
    assert "measurements.relwidth_cwt_integration" in visible
    assert "processing.try_to_use_gpu" not in visible


def test_a_matching_section_keeps_all_of_its_settings(widget):
    """Filtering for a section shows that section, not nothing."""
    _filter(widget, "measurements")

    visible = _visible_settings(widget)
    assert visible == {
        f"measurements.{key}" for key in DEFAULT_SETTINGS["measurements"]
    }


def test_several_words_all_have_to_match(widget):
    _filter(widget, "cells color")

    assert _visible_settings(widget) == {
        "vectorization.cells_edge_color",
        "vectorization.cells_face_color",
        "rasterization.cells_color",
    }


def test_the_filter_opens_the_sections_it_finds_something_in(widget):
    """A hit inside a collapsed section would otherwise stay hidden."""
    from superqt import QCollapsible

    _filter(widget, "spatial_resolution")

    sections = {
        section._toggle_btn.text(): section.isExpanded()
        for section in widget._form.native.findChildren(QCollapsible)
        if section.isVisible()
    }
    assert sections["samples_metadata"]
    assert sections["fields"]
    assert sections["spatial_resolution"]


def test_a_filter_matching_nothing_says_so(widget):
    _filter(widget, "zzz")

    assert _visible_settings(widget) == set()
    assert widget._form._no_match_label.isVisible()


def test_clearing_the_filter_brings_everything_back(widget):
    _filter(widget, "gpu")
    _filter(widget, "")

    assert len(_visible_settings(widget)) == len(
        list(_leaf_entries(widget._form._entries))
    )
    assert not widget._form._no_match_label.isVisible()


def test_clearing_the_filter_restores_the_default_sections(widget):
    """Back to the state of a form that was just built."""
    from superqt import QCollapsible

    def expanded():
        return {
            section._toggle_btn.text(): section.isExpanded()
            for section in widget._form.native.findChildren(QCollapsible)
        }

    before = expanded()
    _filter(widget, "spatial_resolution")
    _filter(widget, "")

    assert expanded() == before


def test_filtering_hides_settings_without_dropping_them(widget):
    """
    The editors of the hidden settings are still read on apply. A filter that
    lost them would write a settings file missing everything not searched for.
    """
    _filter(widget, "gpu")

    assert widget._form.collect() == json.loads(json.dumps(DEFAULT_SETTINGS))


def test_applying_while_filtered_changes_only_what_was_edited(
    widget, settings_file
):
    _filter(widget, "separator")
    _editor_for(widget, "tables", "separator").setText("#")
    widget._apply()

    stored = json.loads(settings_file.read_text())
    expected = deepcopy(DEFAULT_SETTINGS)
    expected["tables"]["separator"] = "#"
    assert stored == json.loads(json.dumps(expected))


def test_a_rebuilt_form_is_filtered_again(widget):
    """
    Reload and reset build a new form, which starts out unfiltered while the
    field still shows what was typed.
    """
    from qtpy.QtWidgets import QApplication

    _filter(widget, "gpu")

    widget._build_form()
    for _ in range(3):  # the widgets of the new form are shown along the way
        QApplication.instance().processEvents()

    assert _visible_settings(widget) == {"processing.try_to_use_gpu"}


# ------------------------------------------------- expanding all the sections


def _sections(widget):
    from superqt import QCollapsible

    return widget._form.native.findChildren(QCollapsible)


def test_the_expand_button_is_above_the_form(widget):
    layout = widget.native.layout()

    assert layout.indexOf(widget._expand_all_button.native) < layout.indexOf(
        widget._form.native
    )


def test_expand_all_opens_every_section_including_nested_ones(widget):
    """
    The form opens with the nested sections closed, so the metadata fields --
    the deepest ones -- are what this has to reach.
    """
    sections = _sections(widget)
    assert not all(section.isExpanded() for section in sections)
    assert any(
        widget._form._nesting_depth(section) > 1 for section in sections
    )

    widget._toggle_all_sections()

    assert all(section.isExpanded() for section in sections)
    assert widget._expand_all_button.text == "Collapse all"


def test_collapse_all_closes_every_section(widget):
    widget._toggle_all_sections()  # expand
    widget._toggle_all_sections()  # collapse

    assert not any(section.isExpanded() for section in _sections(widget))
    assert widget._expand_all_button.text == "Expand all"


def test_the_button_keeps_toggling(widget):
    for expected in ("Collapse all", "Expand all", "Collapse all"):
        widget._toggle_all_sections()
        assert widget._expand_all_button.text == expected


def test_a_rebuilt_form_starts_the_button_over(widget):
    """A reload or a reset builds a form in its default state."""
    widget._toggle_all_sections()
    assert widget._expand_all_button.text == "Collapse all"

    widget._build_form()

    assert widget._expand_all_button.text == "Expand all"
    assert not all(section.isExpanded() for section in _sections(widget))


def test_expanding_everything_does_not_touch_the_values(widget):
    """Opening the sections must not disturb what the editors hold."""
    before = widget._form.collect()

    widget._toggle_all_sections()

    assert widget._form.collect() == before


# ------------------------------------------------- the list editor is a field


def test_every_list_editor_is_marked_as_an_input(widget):
    """
    napari's stylesheet does not cover QPlainTextEdit, so without a style of
    its own a list editor is indistinguishable from a label: same background as
    the form, no border, and nothing that says it can be typed in.
    """
    from qtpy.QtWidgets import QPlainTextEdit

    editors = widget.native.findChildren(QPlainTextEdit)
    assert editors  # the file extensions alone are ten of them

    for editor in editors:
        style = editor.styleSheet()
        assert "border" in style
        assert "background-color" in style
        assert ":focus" in style


def test_a_list_editor_does_not_wrap_its_entries(widget):
    """
    One line is one entry, so a wrapped entry would read as two: in a narrow
    dock ".crossdating" would show up as ".crossdatin" and "g".
    """
    from qtpy.QtWidgets import QPlainTextEdit

    editors = widget.native.findChildren(QPlainTextEdit)
    assert all(
        editor.lineWrapMode() == QPlainTextEdit.NoWrap for editor in editors
    )


def test_list_editor_style_follows_the_theme():
    """The colors come from the theme, not from a hardcoded dark palette."""
    from napari.utils.theme import darken, get_theme

    with patch(
        "napari.settings.get_settings"
    ) as get_napari_settings:
        get_napari_settings.return_value.appearance.theme = "light"
        style = _list_editor_style()

    light = get_theme("light")
    assert darken(light.background, 15) in style
    assert str(light.secondary) in style


def test_list_editor_style_survives_a_theme_it_cannot_read():
    """A plain Qt application has no napari theme; the box must still show."""
    with patch(
        "napari.utils.theme.get_theme", side_effect=RuntimeError("no theme")
    ):
        style = _list_editor_style()

    assert "border" in style


# --------------------------------------------------- opening the file itself


def test_open_settings_file_hands_the_file_to_the_system(
    widget, settings_file
):
    with patch(
        "napari_roxas_ai._settings._settings_widget.QDesktopServices.openUrl",
        return_value=True,
    ) as open_url:
        widget._open_settings_file()

    url = open_url.call_args[0][0]
    assert url.isLocalFile()
    assert url.toLocalFile() == str(settings_file)


def test_open_settings_file_writes_a_missing_file_first(widget, settings_file):
    """Opening a path that does not exist would just fail."""
    settings_file.unlink()

    with patch(
        "napari_roxas_ai._settings._settings_widget.QDesktopServices.openUrl",
        return_value=True,
    ):
        widget._open_settings_file()

    assert json.loads(settings_file.read_text()) == SettingsManager().as_dict()


def test_open_settings_file_reports_a_system_without_an_editor(
    widget, settings_file
):
    """The path is shown, so the file stays reachable by hand."""
    with patch(
        "napari_roxas_ai._settings._settings_widget.QDesktopServices.openUrl",
        return_value=False,
    ), patch(
        "napari_roxas_ai._settings._settings_widget.QMessageBox.warning"
    ) as warning:
        widget._open_settings_file()

    assert warning.called
    assert str(settings_file) in warning.call_args[0][2]


def test_open_settings_file_changes_nothing(widget, settings_file):
    """Looking at the file must not apply the form or rewrite the file."""
    _editor_for(widget, "tables", "separator").setText("#")
    before = settings_file.read_text()

    with patch(
        "napari_roxas_ai._settings._settings_widget.QDesktopServices.openUrl",
        return_value=True,
    ):
        widget._open_settings_file()

    assert settings_file.read_text() == before
    assert SettingsManager().get("tables.separator") == ";"


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


def test_the_open_button_sits_above_the_reset_button(widget):
    layout = widget.native.layout()

    assert layout.indexOf(widget._reload_button.native) < layout.indexOf(
        widget._open_file_button.native
    )
    assert layout.indexOf(widget._open_file_button.native) < layout.indexOf(
        widget._reset_button.native
    )


def test_the_form_takes_the_height_the_buttons_leave(widget):
    """
    The widget is docked at the side of the napari window, so the form has to
    fill it. Its size policy is what both the layout here and napari's
    QtViewerDockWidget read to decide that.
    """
    from qtpy.QtWidgets import QSizePolicy

    assert (
        widget._form.native.sizePolicy().verticalPolicy()
        == QSizePolicy.Expanding
    )

    layout = widget.native.layout()
    buttons = [
        layout.itemAt(index).widget()
        for index in range(layout.count())
        if layout.itemAt(index).widget() is not widget._form.native
    ]
    assert all(
        button.sizePolicy().verticalPolicy() == QSizePolicy.Fixed
        for button in buttons
    )

    widget.native.resize(400, 900)
    widget.native.show()
    assert widget._form.native.height() > sum(
        button.height() for button in buttons
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
