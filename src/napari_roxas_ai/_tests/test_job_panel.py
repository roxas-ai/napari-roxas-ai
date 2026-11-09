from napari_roxas_ai._job_runner import JobHandle, JobProgressPanel
from napari_roxas_ai._jobs import Job


def running_handle() -> JobHandle:
    """A handle of a running job, driven by hand in the tests."""
    return JobHandle(Job("Saving", lambda: None))


def test_cancel_stops_the_job(qtbot):
    panel = JobProgressPanel()
    handle = running_handle()
    panel.track(handle)
    panel._cancel_button.changed.emit(True)
    assert handle.cancel_requested
