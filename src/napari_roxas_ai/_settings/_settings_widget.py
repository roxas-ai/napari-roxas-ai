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
    Tuple,
)

from magicgui.widgets import Container, PushButton
from napari.utils.notifications import show_info
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
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
) -> Callable[[], Dict[str, Any]]:
    """
    Add one row per entry of the mapping to the form, nesting where needed.

    Entries keep the order they have in the settings file, so the form reads
    like the file it edits. Returns a collector that rebuilds the mapping from
    the editors that were created.
    """
    collectors: Dict[str, Callable[[], Any]] = {}

    for key, value in mapping.items():
        if isinstance(value, dict):
            group, nested_form = _make_group(key, expanded=expand_children)
            collectors[key] = _build_group(value, nested_form)
            form.addRow(group)

        elif _is_list_of_dicts(value):
            group, nested_form = _make_group(key, expanded=expand_children)
            collectors[key] = _build_item_list(value, nested_form)
            form.addRow(group)

        else:
            editor, getter = _make_editor(
                value, read_only=key in read_only_keys
            )
            form.addRow(f"{key}:", editor)
            collectors[key] = getter

    return lambda: {key: getter() for key, getter in collectors.items()}


def _build_item_list(
    items: List[Dict[str, Any]], form: QFormLayout
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

    for index, item in enumerate(items):
        title = str(item.get("id") or f"[{index}]")
        group, nested_form = _make_group(title, expanded=False)
        collectors.append(
            _build_group(item, nested_form, read_only_keys=frozenset({"id"}))
        )
        form.addRow(group)

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
        form = QFormLayout()
        outer_layout.addLayout(form)
        # Keeps the rows at the top while the scroll area is taller than them
        outer_layout.addStretch(1)

        self._collect = _build_group(settings, form, expand_children=True)

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
                    form,
                    self._apply_button,
                    self._reload_button,
                    self._open_file_button,
                    self._reset_button,
                ]
            )
        else:
            # Putting the new form back in front keeps the buttons at the bottom
            self.remove(self._form)
            self.insert(0, form)

        self._form = form

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
