from magicgui.widgets import Container, Label, ProgressBar, PushButton

from napari_roxas_ai._jobs import Progress

from ._handle import JobHandle


class JobProgressPanel(Container):
    """
    Shows the progress of one job at a time, with a Cancel button.

    Bind to a job with ``track(handle)``.
    """

    def __init__(self, cancellable: bool = True, **kwargs):
        kwargs.setdefault("labels", False)
        super().__init__(**kwargs)
        self._handle: JobHandle | None = None

        self._label = Label(value="")
        self._bar = ProgressBar(value=0, min=0, max=100, label="Progress")
        self._cancel_button = PushButton(text="Cancel")
        self._cancel_button.visible = cancellable
        self._cancel_button.changed.connect(self.cancel)
        self._row = Container(
            widgets=[self._bar, self._cancel_button],
            layout="horizontal",
            labels=False,
        )
        self._row.margins = (0, 0, 0, 0)
        self.extend([self._label, self._row])
        self.visible = False

    @property
    def handle(self) -> JobHandle | None:
        return self._handle

    @property
    def value(self) -> int:
        return self._bar.value

    @property
    def text(self) -> str:
        return self._label.value

    def track(self, handle: JobHandle | None) -> None:
        """Show the progress of ``handle``."""
        if handle is None:
            return
        self._release()
        if not handle.is_running:
            return
        self._handle = handle
        handle.connect(
            on_progress=self._on_progress, on_finished=self._on_finished
        )
        self._cancel_button.enabled = True
        self._show(handle.progress)
        self.visible = True

    def cancel(self) -> None:
        if self._handle is not None:
            self._cancel_button.enabled = False
            self._label.value = f"{self._handle.name}: cancelling..."
            self._handle.cancel()

    def _on_progress(self, progress: Progress) -> None:
        self._show(progress)

    def _on_finished(self, handle: JobHandle) -> None:
        if handle is self._handle:
            self._release()

    def _show(self, progress: Progress | None) -> None:
        name = self._handle.name if self._handle is not None else ""
        known = progress is not None and progress.total > 0
        if known:
            self._bar.max = 100
            self._bar.value = progress.percent
        else:
            # A bar with max == min == 0 is shown as busy in magicgui
            self._bar.value = 0
            self._bar.max = 0
        detail = ""
        if progress is not None:
            detail = progress.message or (
                f"{progress.current}/{progress.total}" if known else ""
            )
        self._label.value = f"{name}: {detail}" if detail else name

    def _release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.disconnect(
                on_progress=self._on_progress, on_finished=self._on_finished
            )
        self.visible = False
        self._bar.value = 0
        self._bar.max = 100
        self._label.value = ""
