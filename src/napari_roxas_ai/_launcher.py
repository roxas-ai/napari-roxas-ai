"""
Entry point of the `napari-roxas` command.

napari runs no plugin code on startup, so the plugin shortcuts would otherwise
only be installed once a plugin widget is opened. This command starts napari
with the shortcuts installed and the Manual Editing group docked.
"""

import napari

from napari_roxas_ai.shortcuts import install_wasd_shortcuts

PLUGIN_NAME = "napari-roxas-ai"
STARTUP_WIDGET = "Group - Manual Editing"


def main() -> None:
    viewer = napari.Viewer()
    install_wasd_shortcuts(viewer)
    # Docked through the manifest, so napari knows the widget is open and
    # raises it instead of creating a second one when chosen from the menu
    viewer.window.add_plugin_dock_widget(PLUGIN_NAME, STARTUP_WIDGET)
    napari.run()


if __name__ == "__main__":
    main()
