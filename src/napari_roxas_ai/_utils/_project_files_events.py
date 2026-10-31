"""
Notification that files were written to a project directory.

Widgets that list the files of a project (e.g. 3B, 9B) only read the project
directory when they are opened or when the project directory changes. When
another widget writes files to the project (preparation, batch
segmentation), it calls notify_project_files_changed, and the listing widgets
rescan their list. The writing widget does not know who listens.

Callbacks are held weakly, like the settings listeners: a widget that
registers one of its methods is not kept alive by this module, and a callback
whose owner is gone is dropped silently.
"""

import weakref
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

_listeners: List[Any] = []


def add_project_files_listener(callback: Callable[[str], None]) -> None:
    """
    Call `callback(directory)` whenever files were written to a project.

    Args:
        callback: Called with the project directory the files were written to
    """
    if hasattr(callback, "__self__"):
        reference = weakref.WeakMethod(callback)
    else:
        reference = weakref.ref(callback)
    _listeners.append(reference)


def notify_project_files_changed(
    directory: Optional[Union[str, Path]], source: Any = None
) -> None:
    """
    Tell the listeners that files were written to the project `directory`.

    A failing listener is reported and skipped, it must not keep the other
    listeners from being called.

    Args:
        directory: The project directory the files were written to
        source: The widget that wrote the files; its own listener is not
            called, it already knows what it wrote
    """
    if not directory:
        return

    alive = []
    for reference in _listeners:
        callback = reference()
        if callback is None:
            continue
        alive.append(reference)
        if source is not None and getattr(callback, "__self__", None) is source:
            continue
        try:
            callback(str(directory))
        except Exception as e:
            print(f"[project files] Listener failed: {e}")
    _listeners[:] = alive


def is_same_directory(
    first: Optional[Union[str, Path]], second: Optional[Union[str, Path]]
) -> bool:
    """Whether two paths point to the same directory."""
    if not first or not second:
        return False
    return Path(first).resolve() == Path(second).resolve()
