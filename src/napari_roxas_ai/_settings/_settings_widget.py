"""
Widget for viewing and editing the plugin settings.

This replaces the earlier menu entry, which opened settings.json in whatever
text editor the system provides. A hand-edited settings file turned out to be
the main source of breakage: a stray comma, "False" instead of "false", a
Windows path written with single backslashes, or an editor saving the file as
UTF-16 all make the whole file unreadable, and a single mistake costs every
setting in it.

Here each value gets an editor matching its type, so an invalid value cannot be
entered in the first place, and applying the settings writes the file and
updates the running session at the same time -- no restart.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
)

from magicgui.widgets import Container, LineEdit, PushButton
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt, QTimer, QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from ._settings_manager import SettingsManager

if TYPE_CHECKING:
    import napari

# QSpinBox holds 32-bit integers, which is the widest range it can offer
INT_MIN = -2_147_483_648
INT_MAX = 2_147_483_647

# Range offered for float settings: far wider than any of them needs, so that
# the editor never clamps a value the user types
FLOAT_LIMIT = 1e9

# A float editor shows at least MIN_DECIMALS decimals. The floor is what keeps
# the round trip lossless: a QDoubleSpinBox left at its default of 2 decimals
# would turn a spatial resolution of 2.2675 into 2.27 on the first apply.
MIN_DECIMALS = 4
MAX_DECIMALS = 8

# Height of a list editor, in text lines, so that a two-entry file extension
# fits without scrolling and a ten-entry color sequence stays reasonable
MIN_LIST_ROWS = 2
MAX_LIST_ROWS = 12


def _info_icon_style() -> str:
    """Stylesheet that keeps the info icon quieter than the setting name."""
    try:
        from napari.settings import get_settings
        from napari.utils.theme import get_theme

        color = get_theme(get_settings().appearance.theme).secondary
    except Exception:
        color = "gray"

    return f"QLabel {{ color: {color}; }}"


def _list_editor_style() -> str:
    """
    Stylesheet that makes a list editor look like something to type in.

    napari's stylesheet covers QLineEdit and QTextEdit but not the
    QPlainTextEdit used for the lists, which therefore falls through to the
    generic QWidget rule: the background of the form and no border at all, so
    the entries read as a label rather than as an editable field. This gives it
    the fill napari gives a line edit, plus an outline -- a box whose height
    varies with its content needs one more than a single-line field does -- and
    a brighter outline while it has the focus.

    The colors come from the theme in use, so the editor follows a light theme
    and a custom one as the rest of the widget does.
    """
    try:
        from napari.settings import get_settings
        from napari.utils.theme import darken, get_theme

        theme = get_theme(get_settings().appearance.theme)
        # darken(background, 15) is what the stylesheet gives a QLineEdit
        fill = darken(theme.background, 15)
        border, focus_border = theme.secondary, theme.text
    except Exception:
        # No napari theme to read, e.g. in a plain Qt application in a test. A
        # neutral outline still marks the widget as an input either way.
        fill, border, focus_border = "transparent", "gray", "gray"

    return (
        "QPlainTextEdit {"
        f"  background-color: {fill};"
        f"  border: 1px solid {border};"
        "  border-radius: 2px;"
        "  padding: 2px;"
        "}"
        "QPlainTextEdit:focus {"
        f"  border: 1px solid {focus_border};"
        "}"
    )


# What each setting means, by dotted path, shown as the tooltip of its row.
# These are the comments next to the settings in DEFAULT_SETTINGS: keep the two
# in step when adding a setting or changing what one does. A setting that is
# not listed here simply gets no hint.
SETTING_HINTS: Dict[str, str] = {
    "file_extensions.scan_file_extension": "Parts of scan file extension",
    "file_extensions.metadata_file_extension": (
        "Parts of metadata file extension"
    ),
    "file_extensions.cells_file_extension": "Parts of cells file extension",
    "file_extensions.cells_table_file_extension": (
        "And those of the cells table"
    ),
    "file_extensions.rings_file_extension": "Parts of rings file extension",
    "file_extensions.rings_table_file_extension": (
        "And those of the rings table"
    ),
    "file_extensions.crossdating_file_extension": (
        "Parts of tucson file extension"
    ),
    "file_extensions.roxas_file_extensions": "roxas file extensions",
    "file_extensions.image_file_extensions": "Supported image file extensions",
    "file_extensions.text_file_extensions": "Supported text file extensions",
    "JPEG_compression.quality": "Default JPEG quality",
    "JPEG_compression.optimize": "Default optimize flag",
    "JPEG_compression.progressive": "Default progressive flag",
    "processing.try_to_use_gpu": "Try to use GPU if available",
    "processing.try_to_use_autocast": "Try to use autocast if available",
    "vectorization.cells_tolerance": (
        "Default tolerance in pixels for cells vectorization"
    ),
    "vectorization.cells_edge_width": (
        "Default line thickness in pixels for vector shapes visualization"
    ),
    "vectorization.cells_edge_color": (
        "Default color for vector shapes visualization"
    ),
    "vectorization.cells_face_color": (
        "Default color for vector shapes visualization (also used for cells "
        "edition in raster mode)"
    ),
    "vectorization.rings_tolerance": (
        "Default tolerance in pixels for rings vectorization"
    ),
    "vectorization.rings_edge_width": (
        "Default line thickness in pixels for vector shapes visualization"
    ),
    "vectorization.rings_edge_color": (
        "Default color for vector shapes visualization"
    ),
    "vectorization.rerun_interactive_edge_color": (
        "Default color for vector shapes visualization"
    ),
    "measurements.cluster_dbl_cwt_threshold": (
        "Default cluster DBL/CWT threshold in \u00b5m"
    ),
    "measurements.cells_smoothing_kernel_size": (
        "Default smoothing kernel size (1 to disable)"
    ),
    "measurements.relwidth_cwt_integration": (
        "Default wall fraction for thickness measurement"
    ),
    "measurements.cells_tangential_angle": (
        "Default sample angle in degrees (clockwise)"
    ),
    "measurements.lower_limit_cwt_iqr_multiplier": (
        "IQR multiplier for the lower CWT outlier fence"
    ),
    "measurements.upper_limit_cwt_iqr_multiplier": (
        "IQR multiplier for the upper CWT outlier fence"
    ),
    "measurements.opposite_cwt_ratio_limit": (
        "Max CWT ratio between opposite cell sides"
    ),
    "measurements.adjacent_cwt_ratio_limit": (
        "Max CWT ratio between a side and its adjacent sides"
    ),
    "project_directory": "Current project directory",
}

# Marker for a setting that has an explanation. A character rather than an
# icon file: it scales with the font and needs no light and dark variant.
INFO_ICON = "\u24d8"  # circled latin small letter i


class _InfoIcon(QLabel):
    """
    The info icon of a setting: it explains the setting on hover and on click.

    A tooltip alone is easy to miss -- it needs the pointer held still on a
    small target for about a second, which a trackpad makes fiddly and a touch
    screen does not do at all. Clicking shows the same text right away.
    """

    def __init__(self, hint: str):
        super().__init__(INFO_ICON)
        self.setToolTip(hint)
        self.setStyleSheet(_info_icon_style())
        # Says that there is something to click before it is clicked
        self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        """
        Show the explanation, at the icon rather than at the pointer.

        Qt hides the visible tooltip on every mouse press and release, and it
        does so while the click is still being delivered -- showing one from
        inside either handler makes it appear and vanish again. Hence the
        release rather than the press, and the timer on top of it: the tooltip
        is put up once the click is over and nothing is left to take it down.
        """
        QTimer.singleShot(0, self._show_explanation)
        super().mouseReleaseEvent(event)

    def _show_explanation(self) -> None:
        QToolTip.showText(
            self.mapToGlobal(self.rect().center()), self.toolTip(), self
        )


def _row_label(key: str, hint: str) -> QLabel:
    """
    The name of a setting, with the info icon that carries its explanation.

    A setting without an explanation keeps the plain label it had, so the icon
    means "there is something to read here" rather than being decoration on
    every row.
    """
    name = QLabel(f"{key}:")
    if not hint:
        return name

    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    # Keeps the name and its icon against the column of the editors, which is
    # where a form label sits when it is a plain right-aligned QLabel
    layout.addStretch(1)
    layout.addWidget(name)

    icon = _InfoIcon(hint)
    layout.addWidget(icon)

    # On the name and on the row too, so that the explanation shows wherever
    # over the label the pointer ends up (the icon sets its own)
    row.setToolTip(hint)
    name.setToolTip(hint)

    return row


def _decimals_for(value: float) -> int:
    """Number of decimals an editor needs to hold the value without rounding."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        return MAX_DECIMALS
    fraction = text.split(".")[1].rstrip("0") if "." in text else ""
    return min(MAX_DECIMALS, max(MIN_DECIMALS, len(fraction)))


def _make_editor(
    value: Any, read_only: bool = False
) -> Tuple[QWidget, Callable[[], Any]]:
    """
    Build an editor for a single settings value.

    Returns the widget and a getter that returns a value of the same Python type
    as the one passed in. Preserving the type is what makes applying the
    settings a no-op for everything the user did not touch: an int stays an int,
    3.0 does not become 3, and a null stays a null.
    """
    # bool before int, because bool is a subclass of int
    if isinstance(value, bool):
        check_box = QCheckBox()
        check_box.setChecked(value)
        return check_box, check_box.isChecked

    if isinstance(value, int):
        spin_box = QSpinBox()
        spin_box.setRange(INT_MIN, INT_MAX)
        spin_box.setValue(value)
        return spin_box, spin_box.value

    if isinstance(value, float):
        float_spin_box = QDoubleSpinBox()
        float_spin_box.setDecimals(_decimals_for(value))
        float_spin_box.setRange(-FLOAT_LIMIT, FLOAT_LIMIT)
        float_spin_box.setValue(value)
        return float_spin_box, lambda: float(float_spin_box.value())

    if isinstance(value, str):
        line_edit = QLineEdit(value)
        line_edit.setReadOnly(read_only)
        return line_edit, line_edit.text

    if value is None:
        # An empty field means the setting is unset, i.e. null in the file
        unset_line_edit = QLineEdit("")
        unset_line_edit.setPlaceholderText("not set")
        return (
            unset_line_edit,
            lambda: unset_line_edit.text().strip() or None,
        )

    # Only a list of strings gets the line editor. A list holding anything else
    # -- numbers, nested lists, nulls -- would come back as strings, so it is
    # passed through untouched rather than quietly retyped.
    if isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        text_edit = QPlainTextEdit("\n".join(value))
        text_edit.setPlaceholderText("one entry per line")
        text_edit.setStyleSheet(_list_editor_style())
        # One line is one entry, so an entry too long for the box must not be
        # wrapped: in a narrow dock ".crossdating" would look like the two
        # entries ".crossdatin" and "g". It gets a scroll bar instead.
        text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        rows = min(MAX_LIST_ROWS, max(MIN_LIST_ROWS, len(value) + 1))
        text_edit.setFixedHeight(
            rows * text_edit.fontMetrics().lineSpacing() + 12
        )
        return text_edit, lambda: [
            line.strip()
            for line in text_edit.toPlainText().splitlines()
            if line.strip()
        ]

    # No editor for this shape: show it read-only rather than dropping it, so
    # that applying the settings cannot lose a value this widget did not expect
    label = QLabel(str(value))
    label.setEnabled(False)
    return label, lambda: value


def _emptied_file_extensions(settings: Dict[str, Any]) -> str:
    """
    Name of the first file extension setting left empty, or "" if all are set.

    The extensions are consumed as `[0]` when loading and saving samples, so an
    empty list is not a harmless setting but an IndexError the next time a
    sample is touched.
    """
    extensions = settings.get("file_extensions")
    if not isinstance(extensions, dict):
        return ""

    for key, value in extensions.items():
        if isinstance(value, list) and not value:
            return key
    return ""


def _is_list_of_dicts(value: Any) -> bool:
    """True for a list whose entries are all dicts, e.g. the metadata fields."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, dict) for item in value)
    )


class _FormEntry:
    """
    One row of the form -- a setting or a section -- as the filter sees it.

    Built along with the form, because the dotted path of a row is known while
    it is being added and would have to be guessed back out of the widget tree
    afterwards.
    """

    def __init__(
        self,
        path: str,
        widgets: List[QWidget],
        depth: int,
        group: Optional[QCollapsible] = None,
    ):
        # Matching is case insensitive, so the path is stored ready to compare
        self.path = path.lower()
        self.widgets = widgets
        self.depth = depth
        self.group = group
        self.children: List["_FormEntry"] = []

    def set_visible(self, visible: bool) -> None:
        for widget in self.widgets:
            widget.setVisible(visible)

    def show_everything(self) -> int:
        """Show this entry and all of its children, and count the settings."""
        self.set_visible(True)
        if not self.children:
            return 1
        return sum(child.show_everything() for child in self.children)

    def restore_default(self) -> None:
        """Back to the state of a form that was just built."""
        self.set_visible(True)
        if self.group is not None:
            # Only the outermost sections start out open
            if self.depth == 0:
                self.group.expand(animate=False)
            else:
                self.group.collapse(animate=False)
        for child in self.children:
            child.restore_default()

    def apply_filter(self, parts: List[str]) -> int:
        """
        Show what matches every part of the filter, and count the settings.

        A section whose own name matches keeps all of its settings, so that
        filtering for "measurements" shows that section rather than nothing.
        """
        if all(part in self.path for part in parts):
            found = self.show_everything()
            if self.group is not None:
                self.group.expand(animate=False)
            return found

        if not self.children:
            self.set_visible(False)
            return 0

        # Children first: a section is measured for its height when it opens,
        # so its content has to be in its final state by then
        found = sum(child.apply_filter(parts) for child in self.children)
        self.set_visible(bool(found))
        if found and self.group is not None:
            self.group.expand(animate=False)
        return found


def _make_group(
    title: str, expanded: bool
) -> Tuple[QCollapsible, QFormLayout]:
    """Build a collapsible section and return it along with its form layout."""
    body = QWidget()
    form = QFormLayout(body)
    group = QCollapsible(title)
    group.setDuration(0)  # no animation, the sections can be large
    group.addWidget(body)
    if expanded:
        group.expand(animate=False)
    return group, form


def _build_group(
    mapping: Dict[str, Any],
    form: QFormLayout,
    expand_children: bool = False,
    read_only_keys: FrozenSet[str] = frozenset(),
    path: str = "",
    entries: Optional[List[_FormEntry]] = None,
    depth: int = 0,
) -> Callable[[], Dict[str, Any]]:
    """
    Add one row per entry of the mapping to the form, nesting where needed.

    Entries keep the order they have in the settings file, so the form reads
    like the file it edits. Returns a collector that rebuilds the mapping from
    the editors that were created.

    `path` is the dotted path of the mapping itself, which is what the
    explanations of the settings are keyed by. `entries` collects one entry per
    row for the filter to work on.
    """
    collectors: Dict[str, Callable[[], Any]] = {}
    if entries is None:
        entries = []

    for key, value in mapping.items():
        if isinstance(value, dict):
            group, nested_form = _make_group(key, expanded=expand_children)
            entry = _FormEntry(f"{path}{key}", [group], depth, group)
            collectors[key] = _build_group(
                value,
                nested_form,
                path=f"{path}{key}.",
                entries=entry.children,
                depth=depth + 1,
            )
            form.addRow(group)
            entries.append(entry)

        elif _is_list_of_dicts(value):
            group, nested_form = _make_group(key, expanded=expand_children)
            entry = _FormEntry(f"{path}{key}", [group], depth, group)
            collectors[key] = _build_item_list(
                value,
                nested_form,
                path=f"{path}{key}.",
                entries=entry.children,
                depth=depth + 1,
            )
            form.addRow(group)
            entries.append(entry)

        else:
            editor, getter = _make_editor(
                value, read_only=key in read_only_keys
            )
            hint = SETTING_HINTS.get(f"{path}{key}", "")
            label = _row_label(key, hint)
            form.addRow(label, editor)
            collectors[key] = getter
            entries.append(
                _FormEntry(f"{path}{key}", [label, editor], depth)
            )

    return lambda: {key: getter() for key, getter in collectors.items()}


def _build_item_list(
    items: List[Dict[str, Any]],
    form: QFormLayout,
    path: str = "",
    entries: Optional[List[_FormEntry]] = None,
    depth: int = 0,
) -> Callable[[], List[Dict[str, Any]]]:
    """
    Add a list of dictionaries as one collapsible section per entry.

    Entries can be edited, but not added, removed or renamed. The "id" of a
    sample metadata field is the key it gets in a sample's metadata file, so
    changing one is a change of file format rather than of a setting -- and it
    would not even do what it looks like: upgrade_settings() puts the default
    field back under its original id and keeps the renamed one as a user
    addition, so renaming adds a field instead of renaming it. Ids are
    therefore shown read-only.
    """
    collectors = []
    if entries is None:
        entries = []

    for index, item in enumerate(items):
        title = str(item.get("id") or f"[{index}]")
        group, nested_form = _make_group(title, expanded=False)
        entry = _FormEntry(f"{path}{title}", [group], depth, group)
        collectors.append(
            _build_group(
                item,
                nested_form,
                read_only_keys=frozenset({"id"}),
                path=f"{path}{title}.",
                entries=entry.children,
                depth=depth + 1,
            )
        )
        form.addRow(group)
        entries.append(entry)

    return lambda: [collect() for collect in collectors]


class SettingsForm(Container):
    """
    The settings tree as a scrollable form of collapsible sections.

    A Container wrapping a native Qt widget, so that it can sit among the
    magicgui widgets of the settings widget while being built with Qt layouts
    (the same approach as MatplotlibCanvas in the crossdating plotter). napari
    does not put dock widgets in a scroll area, so this brings its own.

    The form declares itself vertically expanding, which is what makes the
    widget fill the dock instead of ending halfway down it. Two things read that
    policy: the layout of the settings widget, which hands all the space left
    over by the buttons to the only item asking for it, and napari, which
    appends a stretch of its own to the bottom of a dock widget unless one of
    its children wants the vertical space (QtViewerDockWidget's
    _maybe_add_vertical_stretch) -- that stretch would otherwise take the room
    the form is meant to get.
    """

    def __init__(self, settings: Dict[str, Any]):
        inner = QWidget()
        outer_layout = QVBoxLayout(inner)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Shown in place of the form when a filter matches nothing, so that an
        # empty form does not read as a broken widget
        self._no_match_label = QLabel("No setting matches the filter.")
        self._no_match_label.setVisible(False)
        outer_layout.addWidget(self._no_match_label)

        form = QFormLayout()
        outer_layout.addLayout(form)
        # Keeps the rows at the top while the scroll area is taller than them
        outer_layout.addStretch(1)

        self._entries: List[_FormEntry] = []
        self._collect = _build_group(
            settings, form, expand_children=True, entries=self._entries
        )

        scroll_area = QScrollArea()
        scroll_area.setWidget(inner)
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # The dock can be dragged to any height, so the form must be able to
        # shrink as far as the scroll bar allows rather than hold a floor
        scroll_area.setMinimumHeight(0)

        super().__init__(widgets=[])
        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)
        self.native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def collect(self) -> Dict[str, Any]:
        """Read every editor back into a settings dictionary."""
        return self._collect()

    def apply_filter(self, text: str) -> int:
        """
        Show only the settings whose path contains what was typed.

        Matching is case insensitive and on part of a name, so "gpu" finds
        try_to_use_gpu. Several words all have to appear, in any order and
        anywhere in the path, which is what makes "cells color" find
        rasterization.cells_color.

        Returns the number of settings left visible.
        """
        parts = text.lower().split()

        if not parts:
            found = sum(entry.show_everything() for entry in self._entries)
            for entry in self._entries:
                entry.restore_default()
        else:
            found = sum(entry.apply_filter(parts) for entry in self._entries)

        self._no_match_label.setVisible(found == 0)
        return found

    def set_all_expanded(self, expanded: bool) -> None:
        """
        Expand or collapse every section, nested ones included.

        A QCollapsible caps the height of its content at the size that content
        has at the moment it is expanded, so the sections are walked deepest
        first: every parent is then measured with its children already open.
        (A parent expanded first is resized by its children afterwards through
        the event filter QCollapsible installs on them, so both orders come out
        right in the end -- this one does not have to rely on that.)
        """
        groups = self.native.findChildren(QCollapsible)
        groups.sort(key=self._nesting_depth, reverse=True)

        for group in groups:
            if expanded:
                group.expand(animate=False)
            else:
                group.collapse(animate=False)

    def _nesting_depth(self, group: QCollapsible) -> int:
        """How many sections the given one sits inside."""
        depth, parent = 0, group.parentWidget()
        while parent is not None and parent is not self.native:
            depth += isinstance(parent, QCollapsible)
            parent = parent.parentWidget()
        return depth


class SettingsWidget(Container):
    """Edit the plugin settings and apply them without restarting napari."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        # Without labels=False, magicgui wraps the form in a labeled widget
        # whose own size policy is the one the layout and napari see, so the
        # form's request for the full height would never reach either of them
        super().__init__(labels=False)
        self._viewer = viewer
        self._settings_manager = SettingsManager()
        self._form = None

        self._filter_field = LineEdit(
            tooltip=(
                "Show only the settings whose name contains this text, e.g. "
                "'gpu' for try_to_use_gpu. Upper and lower case do not matter."
            ),
        )
        self._filter_field.native.setPlaceholderText("Filter settings")
        self._filter_field.changed.connect(self._apply_filter)

        # The form opens with its top-level sections expanded and the nested
        # ones collapsed, which is neither of the two states below, so the
        # button offers the one that reveals everything first
        self._all_expanded = False
        self._expand_all_button = PushButton(
            text="Expand all",
            tooltip="Open every section, nested ones included",
        )
        self._expand_all_button.changed.connect(self._toggle_all_sections)

        self._apply_button = PushButton(
            text="Apply",
            tooltip="Save the settings to file and use them right away",
        )
        self._apply_button.changed.connect(self._apply)

        self._reload_button = PushButton(
            text="Reload from file",
            tooltip="Discard unapplied changes and read the file again",
        )
        self._reload_button.changed.connect(self._reload)

        self._open_file_button = PushButton(
            text="Open settings.json",
            tooltip=(
                "Open the settings file in the system editor. Editing it by "
                "hand is what this widget exists to avoid, so use it to look "
                "at the file or to copy it; press 'Reload from file' "
                "afterwards to pick up what you saved there."
            ),
        )
        self._open_file_button.changed.connect(self._open_settings_file)

        self._reset_button = PushButton(
            text="Reset to defaults",
            tooltip="Replace all settings, your own values included",
        )
        self._reset_button.changed.connect(self._reset)

        self._build_form()

    def _build_form(self) -> None:
        """(Re)build the form from the settings currently in memory."""
        form = SettingsForm(self._settings_manager.as_dict())

        if self._form is None:
            self.extend(
                [
                    self._filter_field,
                    self._expand_all_button,
                    form,
                    self._apply_button,
                    self._reload_button,
                    self._open_file_button,
                    self._reset_button,
                ]
            )
        else:
            # The new form takes the place of the old one, which keeps the
            # expand button above it and the other buttons below it
            index = self.index(self._form)
            self.remove(self._form)
            self.insert(index, form)

        self._form = form

        # A fresh form is back to top-level sections expanded, nested ones
        # collapsed, so the button starts over as well
        self._all_expanded = False
        self._expand_all_button.text = "Expand all"

        # The new form is unfiltered, while the field still shows what it was
        # filtered by, so the filter is put back on
        if self._filter_field.value.strip():
            self._apply_filter()

    def _apply(self) -> None:
        """Store what the form holds, in the file and in the running session."""
        new_settings = self._form.collect()

        emptied = _emptied_file_extensions(new_settings)
        if emptied:
            QMessageBox.warning(
                None,
                "Settings not applied",
                f"The file extension setting '{emptied}' is empty.\n\n"
                "Loading and saving samples reads the first entry of each of "
                "these lists, so an empty one would break both. Enter at "
                "least one entry.",
            )
            return

        extensions_changed = new_settings.get(
            "file_extensions"
        ) != self._settings_manager.get("file_extensions")

        self._settings_manager.replace(new_settings)
        self._refresh_open_widgets()

        show_info(
            "Settings applied and saved to "
            f"{self._settings_manager.settings_file}"
        )

        if extensions_changed:
            # Layer names are built from the extensions when a sample is loaded,
            # but looked up again from the current settings afterwards, so
            # samples loaded under the old extensions no longer match
            show_info(
                "File extensions changed: close and reload your samples so "
                "that their layer names match the new extensions."
            )

    def _reload(self) -> None:
        """Drop unapplied edits and show what the file currently holds."""
        if not self._settings_manager.reload():
            QMessageBox.warning(
                None,
                "Settings not reloaded",
                f"{self._settings_manager.settings_file} is missing or cannot "
                "be read.\n\nThe settings currently in use were kept.",
            )
            return

        self._build_form()
        show_info("Settings reloaded from file")

    def _apply_filter(self) -> None:
        """Show only the settings matching the filter, as it is being typed."""
        self._form.apply_filter(self._filter_field.value)

        # Filtering opens the sections it finds something in, and clearing it
        # closes them again, so either way the toggle starts over
        self._all_expanded = False
        self._expand_all_button.text = "Expand all"

    def _toggle_all_sections(self) -> None:
        """Switch between showing every section and showing none."""
        self._all_expanded = not self._all_expanded
        self._form.set_all_expanded(self._all_expanded)
        self._expand_all_button.text = (
            "Collapse all" if self._all_expanded else "Expand all"
        )

    def _open_settings_file(self) -> None:
        """
        Show the settings file in the editor the system uses for .json files.

        Handing the file to the system rather than editing it here is the whole
        point: what the user saves there is picked up by "Reload from file",
        which runs the same upgrade as a napari start, so a file broken by hand
        is reported rather than silently used.
        """
        settings_file = self._settings_manager.settings_file

        # A settings file only missing because nothing has written it yet:
        # opening a path that does not exist would just fail
        if not settings_file.exists():
            self._settings_manager.save_settings()

        if not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(settings_file))
        ):
            QMessageBox.warning(
                None,
                "Settings file not opened",
                f"{settings_file}\n\nThe system has no application "
                "registered for this file. You can open it yourself from the "
                "path above.",
            )
            return

        show_info(
            f"Opened {settings_file}. Unapplied changes in this widget are not "
            "in the file yet, and changes you save there show up here after "
            "'Reload from file'."
        )

    def _reset(self) -> None:
        """Restore the defaults, discarding every value the user set."""
        confirmation = QMessageBox.question(
            None,
            "Reset settings",
            "Replace all settings by the defaults?\n\n"
            "Every value you changed or added will be lost, including the "
            "project directory.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return

        self._settings_manager.reset()
        self._build_form()
        self._refresh_open_widgets()
        show_info("Settings reset to defaults")

    def refresh_from_settings(self) -> None:
        """Show the settings in memory (part of the refresh protocol)."""
        self._build_form()

    def _refresh_open_widgets(self) -> None:
        """
        Let widgets that are already open pick up the new settings.

        Anything that reads a setting when it needs it follows along by itself,
        because SettingsManager is a singleton that every module shares. Only
        values a widget copied into its own state while it was being built need
        this nudge, so widgets opt in by defining refresh_from_settings().

        Uses the public `dock_widgets` mapping, which hands out the inner widget
        of each dock and unwraps a magicgui one on the way. The private
        `_dock_widgets` would work too, but the viewer given to a plugin widget
        is a PublicOnlyProxy that emits a FutureWarning for every underscore
        attribute, and napari turns warnings into notifications -- so it would
        pop up warning bubbles on every apply.
        """
        window = getattr(self._viewer, "window", None)
        dock_widgets = getattr(window, "dock_widgets", None)
        if not dock_widgets:
            return

        for widget in dock_widgets.values():
            if widget is self:
                continue

            refresh = getattr(widget, "refresh_from_settings", None)
            if not callable(refresh):
                continue

            try:
                refresh()
            except Exception as e:
                # A widget failing to refresh must not fail the apply
                print(
                    f"[settings] Could not refresh "
                    f"{type(widget).__name__}: {e}"
                )
