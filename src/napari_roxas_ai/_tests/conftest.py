import pytest


@pytest.fixture
def sync_runner(monkeypatch):
    """
    Run background jobs inline (in the test thread) for the whole test.

    Widgets submitting jobs then behave synchronously: when the submitting
    call returns, the job and all its callbacks have run.
    """
    from napari_roxas_ai._job_runner import JobRunner, _runner

    runner = JobRunner(synchronous=True)
    monkeypatch.setattr(_runner, "_runner", runner)
    return runner
