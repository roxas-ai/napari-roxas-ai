from napari.utils.notifications import show_error, show_info, show_warning

from napari_roxas_ai._jobs import (
    Job,
    JobStatus,
    JobTag,
    Message,
    as_tags,
    run_job_sync,
)

from ._handle import JobHandle
from ._worker import JobWorker


class JobRunner:
    """
    Runs jobs in superqt worker threads (or inline when ``synchronous``).
    """

    def __init__(self, synchronous: bool = False):
        self.synchronous = synchronous
        self._active: list[JobHandle] = []

    def running(self, tag: JobTag | None = None) -> list[JobHandle]:
        """Currently running jobs."""

        if tag is None:
            return list(self._active)
        as_tags([tag])  # throws if bad tag
        return [h for h in self._active if tag in h.tags]

    def is_running(self, tag: JobTag) -> bool:
        return bool(self.running(tag))

    def submit(
        self,
        job: Job,
        *,
        on_progress=None,
        on_message=None,
        on_partial=None,
        on_result=None,
        on_error=None,
        on_cancelled=None,
        on_finished=None,
        notify_errors: bool = True,
    ) -> JobHandle:
        """
        Start a job and return its handle.

        Callbacks run on the GUI thread (see ``JobHandle.connect``).
        Messages are shown as napari notifications unless ``on_message`` is
        given; failures are always logged and, unless ``notify_errors`` is
        False, shown as an error notification.

        For an ``exclusive`` job, if another job sharing one of its tags is
        running, nothing is started: the callbacks are attached to that
        running job and its handle is returned.
        """
        running = self._running_exclusive(job)
        handle = running or self._new_handle(
            job, notify_errors=notify_errors, show_messages=on_message is None
        )
        handle.connect(
            on_progress=on_progress,
            on_message=on_message,
            on_partial=on_partial,
            on_result=on_result,
            on_error=on_error,
            on_cancelled=on_cancelled,
            on_finished=on_finished,
        )
        if running is not None:
            return running

        self._active.append(handle)
        if self.synchronous:
            self._run_inline(handle)
        else:
            self._start_worker(handle)
        return handle

    def _new_handle(
        self, job: Job, notify_errors: bool, show_messages: bool
    ) -> JobHandle:
        """A handle with the napari notifications as its first callbacks."""
        handle = JobHandle(job)
        if notify_errors:
            handle.connect(
                on_error=lambda exc: show_error(f"{job.name} failed: {exc}")
            )
        if show_messages:
            handle.connect(on_message=_show_message)
        return handle

    def _running_exclusive(self, job: Job) -> JobHandle | None:
        """Return a handle with same tag, if any."""
        if not job.exclusive:
            return None
        for handle in self._active:
            if handle.tags & job.tags:
                return handle
        return None

    def _start_worker(self, handle: JobHandle) -> None:
        worker = JobWorker(handle.job.generator)
        handle._cancel_hook = worker.quit
        worker.yielded.connect(handle._on_event)
        worker.returned.connect(handle._on_succeeded)
        worker.errored.connect(handle._on_failed)
        worker.aborted.connect(handle._on_cancelled)
        worker.finished.connect(lambda: self._on_finished(handle))
        if handle.cancel_requested:
            worker.quit()
        worker.start()

    def _run_inline(self, handle: JobHandle) -> None:
        outcome = run_job_sync(
            handle.job,
            on_event=handle._on_event,
            should_cancel=lambda: handle.cancel_requested,
        )
        if outcome.status is JobStatus.SUCCEEDED:
            handle._on_succeeded(outcome.result)
        elif outcome.status is JobStatus.FAILED:
            handle._on_failed(outcome.error)
        else:
            handle._on_cancelled()
        self._on_finished(handle)

    def _on_finished(self, handle: JobHandle) -> None:
        self._active.remove(handle)
        handle._on_finished()


def _show_message(message: Message) -> None:
    show = show_warning if message.level == "warning" else show_info
    show(message.text)


# -- Singleton runner --

_runner: JobRunner | None = None


def get_runner() -> JobRunner:
    """The shared JobRunner."""
    global _runner
    if _runner is None:
        _runner = JobRunner()
    return _runner
