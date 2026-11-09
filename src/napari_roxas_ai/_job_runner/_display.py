"""
How a job's progress is shown (shared by JobProgressPanel and JobButton).
"""

from napari_roxas_ai._jobs import Progress


def progress_text(name: str, progress: Progress | None) -> str:
    """The progress message, else the job name with the step count."""
    if progress is None:
        return name
    if progress.message:
        return progress.message
    if progress.total > 0:
        return f"{name} {progress.current}/{progress.total}"
    return name


def bar_range(progress: Progress | None) -> tuple[int, int]:
    """
    (maximum, value) of a 0-based progress bar.

    A maximum of 0 shows the bar as busy: used while the total is unknown.
    """
    if progress is None or progress.total <= 0:
        return 0, 0
    return 100, progress.percent
