import threading
from unittest.mock import patch

import pytest

from napari_roxas_ai._job_runner import JobHandle, JobRunner
from napari_roxas_ai._job_runner._enums import Callback
from napari_roxas_ai._jobs import (
    Job,
    JobStatus,
    JobTag,
    Message,
    Partial,
    Progress,
    run_job_sync,
)

RUNNER_MODULE = "napari_roxas_ai._job_runner._runner"


# --------------------------------------------------------------- operations


def counting(n):
    for i in range(n):
        yield Progress(i, n, f"step {i}")
        yield Partial(i)
    yield Message("almost done")
    yield Progress(n, n)
    return n * 10


def failing(exc_type=ValueError):
    yield Progress(0, 1)
    raise exc_type("boom")


def plain(x):
    return x + 1


def endless(started: threading.Event, closed: threading.Event):
    started.set()
    try:
        i = 0
        while True:
            yield Progress(i, 0)
            i += 1
            threading.Event().wait(0.001)
    finally:
        closed.set()


def record_thread(threads):
    threads.append(threading.current_thread())
    yield Progress(1, 1)
    return "ok"


# ------------------------------------------------------------- run_job_sync


def test_run_job_sync_collects_events_and_result():
    outcome = run_job_sync(Job("count", counting, args=(3,)))
    assert outcome.status is JobStatus.SUCCEEDED
    assert outcome.result == 30
    assert outcome.partials == [0, 1, 2]
    assert [m.text for m in outcome.messages] == ["almost done"]
    assert outcome.progress[-1] == Progress(3, 3)


def test_run_job_sync_captures_errors():
    outcome = run_job_sync(Job("fail", failing))
    assert outcome.status is JobStatus.FAILED
    assert isinstance(outcome.error, ValueError)


def test_run_job_sync_accepts_plain_functions():
    outcome = run_job_sync(Job("plain", plain, args=(1,)))
    assert outcome.status is JobStatus.SUCCEEDED
    assert outcome.result == 2


def test_run_job_sync_cancel_closes_generator():
    started, closed = threading.Event(), threading.Event()
    calls = []

    def should_cancel():
        calls.append(1)
        return len(calls) > 3

    outcome = run_job_sync(
        Job("endless", endless, args=(started, closed)),
        should_cancel=should_cancel,
    )
    assert outcome.status is JobStatus.CANCELLED
    assert closed.is_set()


def test_job_normalizes_tags():
    job = Job("x", plain, tags=[JobTag.SAVING, JobTag.LOADING, JobTag.SAVING])
    assert job.tags == frozenset({JobTag.SAVING, JobTag.LOADING})


def test_string_tags_are_rejected():
    with pytest.raises(TypeError, match="JobTag"):
        Job("x", plain, tags={"saving"})
    with pytest.raises(TypeError, match="JobTag"):
        JobRunner(synchronous=True).running("saving")


# ------------------------------------------------------- synchronous runner


def test_sync_runner_callback_order():
    runner = JobRunner(synchronous=True)
    calls = []
    with patch(f"{RUNNER_MODULE}.show_info"):
        handle = runner.submit(
            Job("count", counting, args=(2,)),
            on_progress=lambda p: calls.append(("progress", p.current)),
            on_partial=lambda v: calls.append(("partial", v)),
            on_result=lambda r: calls.append(("result", r)),
            on_finished=lambda h: calls.append(("finished", h.status)),
        )
    assert handle.status is JobStatus.SUCCEEDED
    assert handle.result == 20
    assert calls == [
        ("progress", 0),
        ("partial", 0),
        ("progress", 1),
        ("partial", 1),
        ("progress", 2),
        ("result", 20),
        ("finished", JobStatus.SUCCEEDED),
    ]
    assert runner.running() == []


def test_sync_runner_error_notifies_and_finishes():
    runner = JobRunner(synchronous=True)
    errors, finished = [], []
    with patch(f"{RUNNER_MODULE}.show_error") as show_error:
        handle = runner.submit(
            Job("Failing job", failing),
            on_error=errors.append,
            on_finished=finished.append,
        )
    assert handle.status is JobStatus.FAILED
    assert isinstance(errors[0], ValueError)
    assert finished == [handle]
    show_error.assert_called_once()
    assert "Failing job failed" in show_error.call_args[0][0]


def test_sync_runner_runtime_error_is_a_regular_failure():
    runner = JobRunner(synchronous=True)
    with patch(f"{RUNNER_MODULE}.show_error"):
        handle = runner.submit(Job("fail", failing, args=(RuntimeError,)))
    assert handle.status is JobStatus.FAILED


def test_messages_shown_unless_handled():
    runner = JobRunner(synchronous=True)
    with patch(f"{RUNNER_MODULE}.show_info") as show_info:
        runner.submit(Job("count", counting, args=(1,)))
    show_info.assert_called_once_with("almost done")

    received = []
    with patch(f"{RUNNER_MODULE}.show_info") as show_info:
        runner.submit(
            Job("count", counting, args=(1,)), on_message=received.append
        )
    show_info.assert_not_called()
    assert received == [Message("almost done")]


def raise_key_error(*_):
    raise KeyError("oops")


def test_callback_errors_propagate():
    runner = JobRunner(synchronous=True)
    with (
        patch(f"{RUNNER_MODULE}.show_info"),
        pytest.raises(KeyError, match="oops"),
    ):
        runner.submit(
            Job("count", counting, args=(1,)), on_result=raise_key_error
        )


def test_callback_error_is_not_a_job_failure():
    runner = JobRunner(synchronous=True)
    with (
        patch(f"{RUNNER_MODULE}.show_error") as show_error,
        pytest.raises(KeyError),
    ):
        runner.submit(
            Job("count", counting, args=(1,)), on_progress=raise_key_error
        )
    show_error.assert_not_called()


def test_threaded_callback_error_propagates(qtbot):
    runner = JobRunner()
    finished = []
    with qtbot.capture_exceptions() as exceptions:
        runner.submit(
            Job("plain", plain, args=(1,)),
            on_result=raise_key_error,
            on_finished=finished.append,
        )
        qtbot.waitUntil(lambda: bool(finished), timeout=5000)
    assert [type(e[1]) for e in exceptions] == [KeyError]


def test_connect_after_done_runs_outcome_callbacks():
    runner = JobRunner(synchronous=True)
    with patch(f"{RUNNER_MODULE}.show_info"):
        handle = runner.submit(Job("count", counting, args=(1,)))
    results, finished = [], []
    handle.connect(on_result=results.append, on_finished=finished.append)
    assert results == [10]
    assert finished == [handle]


def test_disconnect_stops_callbacks():
    handle = JobHandle(Job("plain", plain, args=(1,)))
    progress, finished = [], []
    handle.connect(on_progress=progress.append, on_finished=finished.append)
    handle.disconnect(on_finished=finished.append)

    handle._emit(Callback.PROGRESS, Progress(1, 2))
    handle._emit(Callback.FINISHED, handle)
    assert progress == [Progress(1, 2)]
    assert finished == []


def test_callback_can_disconnect_itself():
    handle = JobHandle(Job("plain", plain, args=(1,)))
    calls = []

    def once(p):
        calls.append(p)
        handle.disconnect(on_progress=once)

    handle.connect(on_progress=once)
    handle._emit(Callback.PROGRESS, Progress(0, 2))
    handle._emit(Callback.PROGRESS, Progress(1, 2))
    assert calls == [Progress(0, 2)]


def test_disconnect_unknown_callback_raises():
    runner = JobRunner(synchronous=True)
    handle = runner.submit(Job("plain", plain, args=(1,)))
    with pytest.raises(ValueError):
        handle.disconnect(on_result=print)


# ------------------------------------------------------- threaded runner


def test_threaded_runner_runs_off_main_thread_and_reports_on_main(qtbot):
    runner = JobRunner()
    threads, callback_threads = [], []
    handle = runner.submit(
        Job("thread", record_thread, args=(threads,)),
        on_progress=lambda p: callback_threads.append(
            threading.current_thread()
        ),
        on_result=lambda r: callback_threads.append(
            threading.current_thread()
        ),
    )
    qtbot.waitUntil(lambda: not handle.is_running, timeout=5000)
    assert handle.status is JobStatus.SUCCEEDED
    assert handle.result == "ok"
    assert threads[0] is not threading.main_thread()
    assert callback_threads == [threading.main_thread()] * 2


@pytest.mark.parametrize("exc_type", [ValueError, RuntimeError])
def test_threaded_runner_error_finishes(qtbot, exc_type):
    runner = JobRunner()
    finished = []
    with patch(f"{RUNNER_MODULE}.show_error") as show_error:
        handle = runner.submit(
            Job("fail", failing, args=(exc_type,)), on_finished=finished.append
        )
        qtbot.waitUntil(lambda: bool(finished), timeout=5000)
    assert handle.status is JobStatus.FAILED
    assert isinstance(handle.error, exc_type)
    show_error.assert_called_once()
    assert runner.running() == []


def test_threaded_runner_cancel(qtbot):
    runner = JobRunner()
    started, closed = threading.Event(), threading.Event()
    cancelled, finished = [], []
    handle = runner.submit(
        Job("endless", endless, args=(started, closed)),
        on_cancelled=lambda: cancelled.append(True),
        on_finished=finished.append,
    )
    qtbot.waitUntil(started.is_set, timeout=5000)
    handle.cancel()
    qtbot.waitUntil(lambda: bool(finished), timeout=5000)
    assert handle.status is JobStatus.CANCELLED
    assert cancelled == [True]
    assert closed.is_set()


def test_exclusive_jobs_share_the_running_handle(qtbot):
    runner = JobRunner()
    started, closed = threading.Event(), threading.Event()
    first_finished, second_finished = [], []
    first = runner.submit(
        Job(
            "dl",
            endless,
            args=(started, closed),
            tags={JobTag.RINGS_MODELS_DOWNLOAD},
            exclusive=True,
        ),
        on_finished=first_finished.append,
    )
    second = runner.submit(
        Job(
            "dl",
            plain,
            args=(1,),
            tags={JobTag.RINGS_MODELS_DOWNLOAD},
            exclusive=True,
        ),
        on_finished=second_finished.append,
    )
    assert second is first
    assert runner.running(JobTag.RINGS_MODELS_DOWNLOAD) == [first]
    qtbot.waitUntil(started.is_set, timeout=5000)
    first.cancel()
    qtbot.waitUntil(lambda: bool(second_finished), timeout=5000)
    assert first_finished == second_finished == [first]
    assert not runner.is_running(JobTag.RINGS_MODELS_DOWNLOAD)


def test_exclusive_jobs_with_different_tags_run_side_by_side(qtbot):
    runner = JobRunner()
    events = [threading.Event() for _ in range(4)]
    cells = runner.submit(
        Job(
            "cells models",
            endless,
            args=(events[0], events[1]),
            tags={JobTag.CELLS_MODELS_DOWNLOAD},
            exclusive=True,
        )
    )
    rings = runner.submit(
        Job(
            "rings models",
            endless,
            args=(events[2], events[3]),
            tags={JobTag.RINGS_MODELS_DOWNLOAD},
            exclusive=True,
        )
    )
    assert rings is not cells
    assert runner.running(JobTag.CELLS_MODELS_DOWNLOAD) == [cells]
    assert runner.running(JobTag.RINGS_MODELS_DOWNLOAD) == [rings]
    qtbot.waitUntil(
        lambda: events[0].is_set() and events[2].is_set(), timeout=5000
    )
    cells.cancel()
    rings.cancel()
    qtbot.waitUntil(lambda: not runner.running(), timeout=5000)


def test_sync_runner_fixture_replaces_the_shared_runner(sync_runner):
    from napari_roxas_ai._job_runner import get_runner

    assert get_runner() is sync_runner
    assert sync_runner.synchronous
