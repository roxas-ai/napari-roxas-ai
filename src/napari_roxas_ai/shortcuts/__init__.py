from ._shortcuts import (
    close_all_dock_widgets,
    has_shortcuts_applied,
    install_wasd_shortcuts,
    is_widget_already_present,
    prevent_duplicate_dock_if_already_present,
    load_batch_operations_group,
    load_manual_editing_group,
    mark_shortcuts_applied,
)

__all__ = [
    "install_wasd_shortcuts",
    "has_shortcuts_applied",
    "mark_shortcuts_applied",
    "close_all_dock_widgets",
    "load_manual_editing_group",
    "load_batch_operations_group",
    "is_widget_already_present",
    "prevent_duplicate_dock_if_already_present",
]
