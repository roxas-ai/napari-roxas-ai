"""
Background jobs.
Operations should yield progress and partial answers and return the result.
Operations should be run with the _job_runner.

Operations run in a worker thread. They must only receive and return plain
data (numpy arrays, paths, dicts, ...): never napari layers or viewers, Qt or
magicgui widgets. Copy what the operation needs on the GUI thread before
submitting (e.g. ``layer.as_layer_data_tuple()``), and apply results on the
GUI thread in the callbacks.
"""

from ._enums import JobStatus, JobTag, as_tags
from ._events import Message, Partial, Progress
from ._job import Event, Job, JobOutcome, run_job_sync

__all__ = [
    "Event",
    "Job",
    "JobOutcome",
    "JobStatus",
    "JobTag",
    "Message",
    "Partial",
    "Progress",
    "as_tags",
    "run_job_sync",
]
