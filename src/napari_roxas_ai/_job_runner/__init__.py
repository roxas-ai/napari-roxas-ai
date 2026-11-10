"""
Running background jobs from widgets (Qt).

ex:
handle = get_runner().submit(
    Job("Saving", save_layers, args=(snapshots,), tags={JobTag.SAVING}),
    on_result=self._on_saved,
    on_finished=self._restore_ui,  # always runs
)
self._progress_panel.track(handle)
"""

from ._button import JobButton
from ._handle import JobHandle
from ._panel import JobProgressPanel
from ._runner import JobRunner, get_runner

__all__ = [
    "JobButton",
    "JobHandle",
    "JobProgressPanel",
    "JobRunner",
    "get_runner",
]
