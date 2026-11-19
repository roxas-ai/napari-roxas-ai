from qtpy.QtCore import QThread
from superqt.utils import GeneratorWorker


class JobWorker(GeneratorWorker):
    """
    GeneratorWorker with three fixes for running arbitrary operations.

    - superqt treats a RuntimeError returned from ``work`` as "the worker was
      deleted" and returns without emitting ``errored`` or ``finished``;
      an operation raising RuntimeError would then leave the GUI waiting
      forever. Raising it makes it a regular error.
    - After an abort, the operation's generator is closed in the worker
      thread so its ``finally`` blocks and context managers run right away.
    - Each job runs in a thread of its own instead of superqt's thread pool.
      A pool thread is foreign to Python: torch (pybind11) keeps stale references
      causing crashes on second loads.
    """

    def start(self) -> None:
        thread = _JobThread(self)
        # Kept alive until it finished
        _threads.add(thread)
        thread.finished.connect(lambda: _threads.discard(thread))
        thread.start()

    def work(self):
        try:
            result = super().work()
        finally:
            self._gen.close()
        if isinstance(result, RuntimeError):
            raise result
        return result


class _JobThread(QThread):
    """A thread running one worker."""

    def __init__(self, worker: JobWorker):
        super().__init__()
        self._worker = worker

    def run(self) -> None:
        self._worker.run()


_threads: set[_JobThread] = set()
