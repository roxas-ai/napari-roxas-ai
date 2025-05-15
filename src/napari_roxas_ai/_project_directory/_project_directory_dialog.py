from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog

from napari_roxas_ai._settings import SettingsManager

settings = SettingsManager()


def open_project_directory_dialog():
    """Open a file dialog to select the input directory path."""
    project_directory = settings.get("project_directory")

    new_project_directory = QFileDialog.getExistingDirectory(
        parent=None,
        caption="Select Project Directory",
        directory=project_directory,
    )
    if new_project_directory:
        settings.set("project_directory", new_project_directory)
        show_info(f"Project directory set to: {new_project_directory}")
