import inspect
from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._enums import JobStatus, JobTag, as_tags
from ._events import Message, Partial, Progress

Event = Progress | Message | Partial


@dataclass(frozen=True)
class Job:
    """
    A unit of background work.

    Args:
        name: Human readable name, used in the progress panel and in error
            notifications.
        func: The operation. Usually a generator function yielding
            Progress / Message / Partial events.
        args, kwargs: Arguments for ``func``.
        tags: JobTag members to find the job again
            (``JobRunner.running(tag)``).
        exclusive: When True, submitting the job while another job sharing
            one of its tags is running does not start a second one: the
            running job is returned instead.
    """

    name: str
    func: Callable[..., Any]
    args: tuple = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    tags: Iterable[JobTag] = frozenset()
    exclusive: bool = False

    def __post_init__(self):
        object.__setattr__(self, "tags", as_tags(self.tags))
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))

    def generator(self) -> Generator[Event, None, Any]:
        """
        A generator running the operation.

        This is a generator function: nothing runs before the first
        ``next()``, so the generator can be created in the GUI thread and
        consumed in a worker thread. Plain (non-generator) operations are
        run as a generator without events.
        """
        result = self.func(*self.args, **self.kwargs)
        if inspect.isgenerator(result):
            result = yield from result
        return result


@dataclass
class JobOutcome:
    status: JobStatus
    result: Any = None
    error: BaseException | None = None
    events: list[Event] = field(default_factory=list)

    @property
    def progress(self) -> list[Progress]:
        return [e for e in self.events if isinstance(e, Progress)]

    @property
    def messages(self) -> list[Message]:
        return [e for e in self.events if isinstance(e, Message)]

    @property
    def partials(self) -> list[Any]:
        return [e.payload for e in self.events if isinstance(e, Partial)]


def run_job_sync(
    job: Job,
    *,
    on_event: Callable[[Event], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> JobOutcome:
    """
    Run a job in the current thread.
    Exceptions raised by the operation are not thrown, they are handled by
    the status FAILED. Exceptions raised by ``on_event`` propagate.
    """
    outcome = JobOutcome(status=JobStatus.RUNNING)
    gen = job.generator()
    try:
        while True:
            if should_cancel is not None and should_cancel():
                outcome.status = JobStatus.CANCELLED
                return outcome
            try:
                event = next(gen)
            except StopIteration as stop:
                outcome.status = JobStatus.SUCCEEDED
                outcome.result = stop.value
                return outcome
            # Any error raised by the operation fails the job
            except Exception as exc:  # noqa: BLE001
                outcome.status = JobStatus.FAILED
                outcome.error = exc
                return outcome
            outcome.events.append(event)
            if on_event is not None:
                on_event(event)
    finally:
        gen.close()
