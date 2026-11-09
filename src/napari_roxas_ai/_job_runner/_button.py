"""
JobButton: a push button that turns into a progress bar while its job runs.
"""

from collections.abc import Callable

from magicgui.widgets import Container, ProgressBar, PushButton
from magicgui.widgets.bases import ButtonWidget
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSizePolicy

from napari_roxas_ai._jobs import Progress

from ._display import bar_range, progress_text
from ._handle import JobHandle


class JobButton(Container):
    """
    A push button showing the progress of its job in its own place.

    Pressing it calls ``start``, which starts the job and returns its handle
    (None when nothing was started). The button is then swapped for a
    progress bar (and a Cancel button) of the same height until the job
    finishes, so the layout does not move. ``track(handle)`` does the same
    for a job started elsewhere.

    Exposes the button's ``text``, ``tooltip`` and ``enabled``.
    While a job runs ``enabled`` is False; a value set meanwhile is applied
    to the button, which shows again once the job finished.
    """

    def __init__(
        self,
        *,
        start: Callable[[], JobHandle | None],
        text: str = "",
        tooltip: str | None = None,
        enabled: bool = True,
        cancellable: bool = True,
        **kwargs,
    ):
        # Created before super().__init__, which sets ``enabled`` and
        # ``tooltip`` through the properties below (forwarded to the button)
        self._handle: JobHandle | None = None
        self._start = start
        self._button = PushButton(text=text)
        self._button.changed.connect(self._on_clicked)
        kwargs.setdefault("labels", False)
        super().__init__(tooltip=tooltip, enabled=enabled, **kwargs)
        self.margins = (0, 0, 0, 0)

        self._bar = ProgressBar(value=0, min=0, max=100)
        self._bar.native.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cancel_button = PushButton(
            text="✕", tooltip="Cancel", visible=cancellable
        )
        self._cancel_button.changed.connect(self.cancel)
        self._progress_row = Container(
            widgets=[self._bar, self._cancel_button],
            layout="horizontal",
            labels=False,
            visible=False,
        )
        self._progress_row.margins = (0, 0, 0, 0)
        # Takes the button's width: the layout must not grow for it
        self._progress_row.native.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.extend([self._button, self._progress_row])

    # -- Forward button properties --

    @property
    def enabled(self) -> bool:
        return self._button.enabled and not self.is_busy

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._button.enabled = value

    @property
    def text(self) -> str:
        return self._button.text

    @text.setter
    def text(self, value: str) -> None:
        self._button.text = value

    @property
    def tooltip(self) -> str | None:
        return self._button.tooltip

    @tooltip.setter
    def tooltip(self, value: str | None) -> None:
        self._button.tooltip = value

    # -- Handling

    @property
    def handle(self) -> JobHandle | None:
        return self._handle

    @property
    def is_busy(self) -> bool:
        return self._handle is not None

    @property
    def progress_text(self) -> str:
        return self._bar.native.format().replace("%%", "%")

    def track(self, handle: JobHandle | None) -> None:
        if handle is None or not handle.is_running:
            return
        self._release()
        self._handle = handle
        handle.connect(
            on_progress=self._on_progress, on_finished=self._on_finished
        )

        # Same height as the button: the layout does not move. The bar
        # fills it (napari's theme makes bars thinner than buttons) and the
        # cancel button is a square
        height = self._button.native.height()
        self._progress_row.native.setFixedHeight(height)
        self._bar.native.setFixedHeight(height)
        self._cancel_button.native.setFixedSize(height, height)
        self._cancel_button.enabled = True
        self._show(handle.progress)
        self._button.visible = False
        self._progress_row.visible = True

    def _on_clicked(self) -> None:
        self.track(self._start())

    def cancel(self) -> None:
        if self._handle is not None:
            self._cancel_button.enabled = False
            self._set_bar_text("Stopping…")
            self._handle.cancel()

    def _on_progress(self, progress: Progress) -> None:
        self._show(progress)

    def _on_finished(self, handle: JobHandle) -> None:
        if handle is self._handle:
            self._release()

    def _show(self, progress: Progress | None) -> None:
        maximum, value = bar_range(progress)
        self._bar.native.setRange(0, maximum)
        self._bar.native.setValue(value)
        self._set_bar_text(progress_text(self._handle.name, progress))

    def _set_bar_text(self, text: str) -> None:
        # "%" starts a placeholder (e.g. %p) in a progress bar format
        self._bar.native.setFormat(text.replace("%", "%%"))

    def _release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        handle.disconnect(
            on_progress=self._on_progress, on_finished=self._on_finished
        )
        self._progress_row.visible = False
        self._button.visible = True


# Laid out like a button: magicgui wraps any other widget of a labelled
# container with a label, which would indent the JobButton
ButtonWidget.register(JobButton)
