from superqt.utils import GeneratorWorker


class JobWorker(GeneratorWorker):
    """
    GeneratorWorker with two fixes for running arbitrary operations.

    - superqt treats a RuntimeError returned from ``work`` as "the worker was
      deleted" and returns without emitting ``errored`` or ``finished``;
      an operation raising RuntimeError would then leave the GUI waiting
      forever. Raising it makes it a regular error.
    - After an abort, the operation's generator is closed in the worker
      thread so its ``finally`` blocks and context managers run right away.
    """

    def work(self):
        try:
            result = super().work()
        finally:
            self._gen.close()
        if isinstance(result, RuntimeError):
            raise result
        return result
