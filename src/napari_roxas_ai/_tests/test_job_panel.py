import threading

from napari_roxas_ai._job_runner import JobProgressPanel, JobRunner
from napari_roxas_ai._jobs import Job, JobStatus, Progress


def steps(gate: threading.Event, n=4):
    for i in range(n):
        yield Progress(i, n, f"file {i}")
        gate.wait(5)
    return n


def endless(started: threading.Event):
    started.set()
    while True:
        yield Progress(0, 0)
        threading.Event().wait(0.001)


def test_panel_hidden_when_idle(qtbot):
    panel = JobProgressPanel()
    assert not panel.visible
    assert panel.handle is None
    panel.track(None)
    assert not panel.visible


def test_panel_tracks_progress_and_hides_on_finish(qtbot):
    runner = JobRunner()
    panel = JobProgressPanel()
    gate = threading.Event()
    handle = runner.submit(Job("Saving", steps, args=(gate,)))
    panel.track(handle)
    assert panel.visible
    assert panel.handle is handle

    qtbot.waitUntil(
        lambda: panel.text.startswith("Saving: file"), timeout=5000
    )
    gate.set()
    qtbot.waitUntil(lambda: not handle.is_running, timeout=5000)
    assert handle.status is JobStatus.SUCCEEDED
    assert not panel.visible
    assert panel.handle is None


def test_panel_cancel_button_cancels_job(qtbot):
    runner = JobRunner()
    panel = JobProgressPanel()
    started = threading.Event()
    handle = runner.submit(Job("Endless", endless, args=(started,)))
    panel.track(handle)
    qtbot.waitUntil(started.is_set, timeout=5000)

    panel._cancel_button.changed.emit(True)
    qtbot.waitUntil(lambda: not handle.is_running, timeout=5000)
    assert handle.status is JobStatus.CANCELLED
    assert not panel.visible


def test_panel_ignores_finished_handle(sync_runner):
    panel = JobProgressPanel()
    handle = sync_runner.submit(Job("Done", steps, args=(_open_gate(), 1)))
    panel.track(handle)
    assert not panel.visible


def _open_gate() -> threading.Event:
    gate = threading.Event()
    gate.set()
    return gate


def test_panel_shows_busy_when_total_is_unknown(qtbot):
    panel = JobProgressPanel()
    panel._show(Progress(3, 10))
    assert (panel._bar.max, panel.value) == (100, 30)

    panel._show(Progress(0, 0, "Loading models"))
    assert (panel._bar.max, panel.value) == (0, 0)
    assert panel.text.endswith("Loading models")

    panel._release()
    assert panel._bar.max == 100
