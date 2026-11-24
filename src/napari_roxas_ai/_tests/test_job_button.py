import pytest

from napari_roxas_ai._job_runner import JobButton, JobHandle
from napari_roxas_ai._jobs import Job


def running_handle(name="Saving") -> JobHandle:
    """A handle of a running job, driven by hand in the tests."""
    return JobHandle(Job(name, lambda: None))


def finish(handle: JobHandle) -> None:
    handle._on_succeeded(None)
    handle._on_finished()


@pytest.fixture
def started():
    """The handles returned by the button's ``start``, one per press."""
    return []


@pytest.fixture
def button(qtbot, started):
    def start():
        handle = running_handle()
        started.append(handle)
        return handle

    return JobButton(text="Save All Layers", start=start)


def press(button: JobButton) -> None:
    button._button.native.click()


def test_pressing_starts_the_job_and_locks_the_button(button, started):
    press(button)
    assert len(started) == 1
    assert button.handle is started[0]
    assert button.is_busy
    assert not button.enabled


def test_button_is_usable_again_when_the_job_finished(button, started):
    press(button)
    finish(started[0])

    assert not button.is_busy
    assert button.enabled
    press(button)
    assert len(started) == 2


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_button_is_usable_again_after_a_failed_or_cancelled_job(
    button, started, outcome
):
    press(button)
    handle = started[0]
    if outcome == "failed":
        handle._on_failed(ValueError("boom"))
    else:
        handle._on_cancelled()
    handle._on_finished()
    assert not button.is_busy
    assert button.enabled


def test_nothing_started_leaves_the_button_usable(qtbot):
    button = JobButton(text="Save", start=lambda: None)
    press(button)
    assert not button.is_busy
    assert button.enabled


def test_enabled_set_while_busy_applies_after_the_job(button, started):
    press(button)
    button.enabled = True  # does not end the busy state
    assert not button.enabled
    button.enabled = False
    finish(started[0])
    assert not button.enabled  # the last value set is kept


def test_disabled_at_creation_stays_disabled(qtbot):
    button = JobButton(text="Run", start=lambda: None, enabled=False)
    assert not button.enabled


def test_cancel_stops_the_job(button, started):
    press(button)
    button._cancel_button.native.click()
    assert started[0].cancel_requested
    # Cannot be cancelled twice
    assert not button._cancel_button.enabled


def test_tracks_a_job_started_elsewhere(button):
    handle = running_handle()
    button.track(handle)
    assert button.is_busy
    finish(handle)
    assert button.enabled


def test_finished_or_missing_handle_is_ignored(button):
    button.track(None)
    handle = running_handle()
    finish(handle)
    button.track(handle)
    assert not button.is_busy
    assert button.enabled
