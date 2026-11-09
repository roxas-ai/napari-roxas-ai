import logging
from collections.abc import Callable
from typing import Any

from napari_roxas_ai._jobs import (
    Event,
    Job,
    JobStatus,
    JobTag,
    Message,
    Partial,
    Progress,
)

from ._enums import Callback

logger = logging.getLogger(__name__)


class JobHandle:
    """
    A submitted job and its state.
    Register to callbacks with ```connect()```, unregister with
    ```disconnect()```. Callbacks are called on the GUI thread.

    The handle owns the job's state transitions (the ``_on_*`` methods): the
    runner only feeds them what the job does.
    """

    def __init__(self, job: Job):
        self.job = job
        self.status = JobStatus.RUNNING
        self.progress: Progress | None = None
        self.result: Any = None
        self.error: BaseException | None = None
        self._cancel_requested = False
        # Set by the runner: stops the running job at its next yield
        self._cancel_hook: Callable[[], None] | None = None
        self._callbacks: dict[Callback, list[Callable]] = {
            kind: [] for kind in Callback
        }

    def __repr__(self) -> str:
        return f"<JobHandle {self.job.name!r} {self.status.value}>"

    @property
    def name(self) -> str:
        return self.job.name

    @property
    def tags(self) -> frozenset[JobTag]:
        return self.job.tags

    @property
    def is_running(self) -> bool:
        return not self.status.is_done

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def cancel(self) -> None:
        """Ask the job to stop at its next yield."""
        if not self.is_running:
            return
        self._cancel_requested = True
        if self._cancel_hook is not None:
            self._cancel_hook()

    def connect(
        self,
        *,
        on_progress: Callable[[Progress], None] | None = None,
        on_message: Callable[[Message], None] | None = None,
        on_partial: Callable[[Any], None] | None = None,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_finished: Callable[["JobHandle"], None] | None = None,
    ) -> "JobHandle":
        """
        Register callbacks.

        ``on_finished`` always runs last, whatever the outcome of the job.
        If the job is already done, the callbacks matching its outcome run immediately.
        """
        given = _by_kind(
            on_progress,
            on_message,
            on_partial,
            on_result,
            on_error,
            on_cancelled,
            on_finished,
        )
        for kind, callback in given.items():
            self._callbacks[kind].append(callback)

        # immediately run callbacks if the job is done already.
        if self.status.is_done:
            if self.status is JobStatus.SUCCEEDED and on_result is not None:
                on_result(self.result)
            elif self.status is JobStatus.FAILED and on_error is not None:
                on_error(self.error)
            elif (
                self.status is JobStatus.CANCELLED and on_cancelled is not None
            ):
                on_cancelled()
            if on_finished is not None:
                on_finished(self)
        return self

    def disconnect(
        self,
        *,
        on_progress: Callable[[Progress], None] | None = None,
        on_message: Callable[[Message], None] | None = None,
        on_partial: Callable[[Any], None] | None = None,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_finished: Callable[["JobHandle"], None] | None = None,
    ) -> None:
        """
        Unregister callbacks registered with ``connect()``.
        """
        given = _by_kind(
            on_progress,
            on_message,
            on_partial,
            on_result,
            on_error,
            on_cancelled,
            on_finished,
        )
        for kind, callback in given.items():
            self._callbacks[kind].remove(callback)

    # -- State transitions, called by the runner on the GUI thread --
    # The state is updated before the callbacks run, so callbacks see it.

    def _on_event(self, event: Event) -> None:
        if isinstance(event, Progress):
            self.progress = event
            self._emit(Callback.PROGRESS, event)
        elif isinstance(event, Message):
            self._emit(Callback.MESSAGE, event)
        elif isinstance(event, Partial):
            self._emit(Callback.PARTIAL, event.payload)
        else:
            logger.warning(
                "Job %r yielded an unsupported value %r; operations must "
                "yield Progress, Message or Partial events.",
                self.name,
                event,
            )

    def _on_succeeded(self, result: Any) -> None:
        self.status = JobStatus.SUCCEEDED
        self.result = result
        self._emit(Callback.RESULT, result)

    def _on_failed(self, error: BaseException) -> None:
        self.status = JobStatus.FAILED
        self.error = error
        logger.error(
            "Job %r failed",
            self.name,
            exc_info=(type(error), error, error.__traceback__),
        )
        self._emit(Callback.ERROR, error)

    def _on_cancelled(self) -> None:
        self.status = JobStatus.CANCELLED
        self._emit(Callback.CANCELLED)

    def _on_finished(self) -> None:
        """Last transition, whatever the outcome: on_finished runs last."""
        self._cancel_hook = None
        if not self.status.is_done:
            # superqt does not emit ``returned`` when the abort is requested
            # after the last yield: the job did finish, but was asked to stop
            self._on_cancelled()
        self._emit(Callback.FINISHED, self)

    def _emit(self, kind: Callback, *args) -> None:
        # Iterate over a copy: callbacks may disconnect themselves
        for callback in list(self._callbacks[kind]):
            callback(*args)


def _by_kind(
    on_progress,
    on_message,
    on_partial,
    on_result,
    on_error,
    on_cancelled,
    on_finished,
) -> dict[Callback, Callable]:
    """The given (not None) callbacks, by kind."""
    given = {
        Callback.PROGRESS: on_progress,
        Callback.MESSAGE: on_message,
        Callback.PARTIAL: on_partial,
        Callback.RESULT: on_result,
        Callback.ERROR: on_error,
        Callback.CANCELLED: on_cancelled,
        Callback.FINISHED: on_finished,
    }
    return {kind: cb for kind, cb in given.items() if cb is not None}
