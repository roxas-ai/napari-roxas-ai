"""
Handles the selection and processing of crossdating files.
"""

import glob
import os
from typing import List, Optional, Tuple

import pandas as pd
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from .._reader._crossdating_reader import read_crossdating_file


class CrossdatingSelectionDialog(QDialog):
    """
    Dialog for selecting crossdating files to process.
    """

    def __init__(
        self,
        project_directory: str,
        text_file_extensions: List[str],
        project_file_path: str,
        parent=None,
    ):
        """
        Initialize the crossdating selection dialog.

        Parameters
        ----------
        project_directory : str
            The project directory containing crossdating files
        text_file_extensions : List[str]
            List of file extensions to consider as text files
        project_file_path : str
            Path to the project crossdating file (to exclude from selection)
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.project_directory = project_directory
        self.text_file_extensions = text_file_extensions
        self.project_file_path = project_file_path
        self.selected_files = []

        self.setWindowTitle("Select Crossdating Files")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._create_ui()
        self._populate_file_list()

    def _create_ui(self):
        """Create the dialog UI components."""
        layout = QVBoxLayout()

        # Instruction label
        info_label = QLabel(
            "Select crossdating files to include in the project. "
            "These files will be merged with the project crossdating file."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # File list widget
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.file_list)

        # Select/Deselect all checkbox
        self.select_all_checkbox = QCheckBox("Select All Files")
        self.select_all_checkbox.stateChanged.connect(self._toggle_select_all)
        layout.addWidget(self.select_all_checkbox)

        # Buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._ok_clicked)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        # Add buttons to layout
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _populate_file_list(self):
        """Find and populate the list with available text files."""
        text_files = []

        # Find all text files in the project directory (including subdirectories)
        for ext in self.text_file_extensions:
            text_files.extend(
                glob.glob(
                    f"{self.project_directory}/**/*{ext}", recursive=True
                )
            )
            text_files.extend(
                glob.glob(
                    f"{self.project_directory}/**/*{ext.upper()}",
                    recursive=True,
                )
            )

        # Exclude the project crossdating file
        if self.project_file_path in text_files:
            text_files.remove(self.project_file_path)

        # Sort files for consistent display
        text_files = sorted(text_files)

        # Add files to the list widget
        for file_path in text_files:
            # Display relative path for better readability
            try:
                display_name = os.path.relpath(
                    file_path, self.project_directory
                )
            except ValueError:
                display_name = os.path.basename(file_path)

            self.file_list.addItem(display_name)
            # Store the full path as item data
            item = self.file_list.item(self.file_list.count() - 1)
            item.setData(1, file_path)

    def _toggle_select_all(self, state):
        """
        Toggle selection of all items in the list.

        Parameters
        ----------
        state : int
            State of the checkbox (0: unchecked, 2: checked)
        """
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setSelected(state == 2)  # Qt.Checked is 2

    def _ok_clicked(self):
        """Handle OK button click - collect selected files."""
        self.selected_files = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.isSelected():
                self.selected_files.append(
                    item.data(1)
                )  # Get full path from item data

        self.accept()

    def get_selected_files(self) -> List[str]:
        """Return the list of selected file paths."""
        return self.selected_files


def process_crossdating_files(
    project_directory: str,
    crossdating_file_extension: str,
    text_file_extensions: List[str],
) -> Optional[pd.DataFrame]:
    """
    Process crossdating files by creating or updating a project crossdating file.

    Parameters
    ----------
    project_directory : str
        The project directory
    crossdating_file_extension : str
        The file extension for crossdating files
    text_file_extensions : List[str]
        List of file extensions to consider as text files

    Returns
    -------
    Optional[pd.DataFrame]
        The merged DataFrame if successful, None otherwise
    """
    # Determine the path for the project crossdating file (always in root directory)
    crossdating_file_path = os.path.join(
        project_directory, f"rings_series{crossdating_file_extension}"
    )

    # Create the file if it doesn't exist
    if not os.path.exists(crossdating_file_path):
        # Create an empty dataframe and save it with tab separator
        pd.DataFrame().to_csv(crossdating_file_path, sep="\t", index=True)

    # Show dialog to select crossdating files
    dialog = CrossdatingSelectionDialog(
        project_directory=project_directory,
        text_file_extensions=text_file_extensions,
        project_file_path=crossdating_file_path,
    )

    result = dialog.exec_()
    if not result:
        # User canceled
        return None

    selected_files = dialog.get_selected_files()
    if not selected_files:
        # No files selected
        return None

    # Process selected files and merge with project file
    merged_df = merge_crossdating_files(selected_files, crossdating_file_path)

    # Save merged data back to the project file
    if merged_df is not None and not merged_df.empty:
        # Use tab separator for saving the file
        merged_df.to_csv(crossdating_file_path, sep="\t", index=True)

    return merged_df


def _try_read_dataframe(filepath: str) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Try to read a DataFrame from a file with different methods.

    Parameters
    ----------
    filepath : str
        Path to the file to read

    Returns
    -------
    Tuple[bool, Optional[pd.DataFrame]]
        Success flag and DataFrame if successful, None otherwise
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False, None

    # First try to read with tab separator
    try:
        df = pd.read_csv(filepath, sep="\t", index_col=0)
        if not df.empty:
            return True, df
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        pass

    # Then try with the more flexible reader
    try:
        df = read_crossdating_file(filepath)
        if not df.empty:
            return True, df
    except (ValueError, FileNotFoundError):
        pass

    return False, None


def merge_crossdating_files(
    source_files: List[str], target_file: str
) -> Optional[pd.DataFrame]:
    """
    Read and merge multiple crossdating files.

    Parameters
    ----------
    source_files : List[str]
        List of source crossdating files to merge
    target_file : str
        Target crossdating file to merge with

    Returns
    -------
    Optional[pd.DataFrame]
        The merged DataFrame if successful, None otherwise
    """
    dfs = []
    error_files = []

    # Read the target file
    success, target_df = _try_read_dataframe(target_file)
    if success and target_df is not None:
        dfs.append(target_df)

    # Read all source files
    for file_path in source_files:
        success, source_df = _try_read_dataframe(file_path)
        if success and source_df is not None:
            dfs.append(source_df)
        else:
            error_files.append(os.path.basename(file_path))
            print(f"Error reading file {file_path}")

    if not dfs:
        return None

    # Different approach to merging to prevent data loss
    if len(dfs) == 1:
        # If only one dataframe, just use it
        merged_df = dfs[0]
    else:
        # Handle potential mixed index types (int, str)
        # Convert all indices to strings for comparison
        for i in range(len(dfs)):
            # Convert index to string if not already
            if not all(isinstance(idx, str) for idx in dfs[i].index):
                dfs[i].index = dfs[i].index.astype(str)

        # Start with an empty dataframe with all possible years from all files
        # Use string representation for all indices to avoid type comparison issues
        all_years = sorted(set().union(*[df.index for df in dfs]))
        all_series = sorted(set().union(*[df.columns for df in dfs]))

        # Create an empty dataframe with all years and all series
        merged_df = pd.DataFrame(index=all_years, columns=all_series)

        # Fill the dataframe with values from each source
        for df in dfs:
            for series in df.columns:
                # Only update non-NaN values
                series_data = df[series].dropna()
                if not series_data.empty:
                    # For each year in this series, update the merged dataframe
                    # only if the value in the merged dataframe is NaN
                    for year, value in series_data.items():
                        if pd.isna(merged_df.at[year, series]):
                            merged_df.at[year, series] = value

    # Report any files that couldn't be read
    if error_files:
        print(
            f"Warning: Could not read the following files: {', '.join(error_files)}"
        )

    return merged_df
