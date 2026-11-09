from enum import Enum, auto


class Callback(Enum):
    """The kinds of callbacks a JobHandle calls (see ``JobHandle.connect``)."""

    PROGRESS = auto()
    MESSAGE = auto()
    PARTIAL = auto()
    RESULT = auto()
    ERROR = auto()
    CANCELLED = auto()
    FINISHED = auto()
