"""
Handles the selection and processing of crossdating files.
"""

import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from napari_roxas_ai._reader._crossdating_reader import read_crossdating_file


class CrossdatingSelectionDialog(QDialog):
    """
    Dialog for selecting crossdating files to process and their scaling.
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
        self.selected_scaling = 10.0  # Default to 1/100 mm (10 um)

        self.setWindowTitle("Prepare Crossdating Files")
        self.setMinimumWidth(500)
        self.setMinimumHeight(450)

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

        # Scaling selection
        scaling_layout = QHBoxLayout()
        scaling_label = QLabel("Units of selected cross-dating files:")
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["1 / 10 mm", "1 / 100 mm", "1 / 1000 mm", "divide values by 10"])
        self.scaling_combo.setCurrentText("1 / 100 mm")
        scaling_layout.addWidget(scaling_label)
        scaling_layout.addWidget(self.scaling_combo)
        layout.addLayout(scaling_layout)

        # Buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._ok_clicked)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        # Add buttons to layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _populate_file_list(self):
        """Find and populate the list with available text files."""
        text_files = set()

        # Find all text files in the project directory (including subdirectories)
        for ext in self.text_file_extensions:
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                found_files = glob.glob(
                    str(Path(self.project_directory) / "**" / pattern),
                    recursive=True,
                )
                for f in found_files:
                    text_files.add(str(Path(f).resolve()))

        # Exclude the project crossdating file
        project_file_path = str(Path(self.project_file_path).resolve())
        if project_file_path in text_files:
            text_files.remove(project_file_path)

        # Sort files for consistent display
        sorted_text_files = sorted(list(text_files))

        # Add files to the list widget
        for file_path in sorted_text_files:
            # Display relative path for better readability
            try:
                display_name = str(
                    Path(file_path).relative_to(self.project_directory)
                )
            except ValueError:
                display_name = Path(file_path).name

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

        # Get scaling factor (target is micrometers)
        scaling_text = self.scaling_combo.currentText()
        if scaling_text == "1 / 10 mm":
            self.selected_scaling = 100.0  # 0.1 mm = 100 um
        elif scaling_text == "1 / 100 mm":
            self.selected_scaling = 10.0   # 0.01 mm = 10 um
        elif scaling_text == "divide values by 10":
            self.selected_scaling = 0.1    # divide values by 10
        else:  # 1 / 1000 mm
            self.selected_scaling = 1.0    # 0.001 mm = 1 um

        self.accept()

    def get_selected_files(self) -> List[str]:
        """Return the list of selected file paths."""
        return self.selected_files

    def get_selected_scaling(self) -> float:
        """Return the selected scaling factor to micrometers."""
        return self.selected_scaling


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
    crossdating_file_path = (
        Path(project_directory) / f"rings_series{crossdating_file_extension}"
    )

    # Create the file if it doesn't exist
    if not crossdating_file_path.exists():
        # Create an empty dataframe and save it with tab separator
        pd.DataFrame().to_csv(str(crossdating_file_path), sep="\t", index=True)

    # Show dialog to select crossdating files
    dialog = CrossdatingSelectionDialog(
        project_directory=project_directory,
        text_file_extensions=text_file_extensions,
        project_file_path=str(crossdating_file_path),
    )

    result = dialog.exec_()
    if not result:
        # User canceled
        return None

    selected_files = dialog.get_selected_files()
    if not selected_files:
        # No files selected
        return None

    scaling_factor = dialog.get_selected_scaling()

    # Process selected files and merge with project file
    merged_df = merge_crossdating_files(
        selected_files, str(crossdating_file_path), scaling_factor
    )

    # Save merged data back to the project file
    if merged_df is not None and not merged_df.empty:
        # Use tab separator for saving the file
        merged_df.to_csv(str(crossdating_file_path), sep="\t", index=True)

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
    if not Path(filepath).exists() or Path(filepath).stat().st_size == 0:
        return False, None

    # Then try with the more flexible reader
    try:
        df = read_crossdating_file(filepath)
        if not df.empty:
            return True, df
    except (ValueError, FileNotFoundError):
        pass

    return False, None


def merge_crossdating_files(
    source_files: List[str], target_file: str, scaling_factor: float = 1.0
) -> Optional[pd.DataFrame]:
    """
    Read and merge multiple crossdating files to replace the existing target.

    Parameters
    ----------
    source_files : List[str]
        List of source crossdating files to process and merge
    target_file : str
        Target crossdating file (will be replaced by new data)
    scaling_factor : float
        Scaling factor to apply to source files to convert them to micrometers.
        Defaults to 1.0 (no scaling).

    Returns
    -------
    Optional[pd.DataFrame]
        The merged DataFrame if successful, None otherwise
    """
    dfs_to_scale = []
    error_files = []

    # Read and scale source files
    for file_path in source_files:
        success, source_df = _try_read_dataframe(file_path)
        if success and source_df is not None:
            # Apply scaling to numeric columns (excluding year-like columns)
            if scaling_factor != 1.0:
                # Ensure it's numeric
                for col in source_df.columns:
                    source_df[col] = pd.to_numeric(source_df[col], errors='coerce')
                
                numeric_cols = source_df.select_dtypes(include=[np.number]).columns
                series_cols = [
                    col
                    for col in numeric_cols
                    if "year" not in str(col).lower() and "index" not in str(col).lower()
                ]
                source_df[series_cols] = source_df[series_cols] * scaling_factor
            
            dfs_to_scale.append(source_df)
        else:
            error_files.append(Path(file_path).name)
            print(f"Error reading file {file_path}")

    if not dfs_to_scale:
        return None

    # Merge everything into a fresh DataFrame
    # All DataFrames to combine (only from source files)
    all_dfs = dfs_to_scale

    # Collect all years and series names from selected files
    all_years_raw = set().union(*[df.index for df in all_dfs])
    all_years = []
    for y in all_years_raw:
        if pd.isna(y): continue
        try:
            all_years.append(int(float(y)))
        except (ValueError, TypeError):
            all_years.append(str(y))
    all_years = list(set(all_years))
    # Sort all_years to ensure numeric sorting if possible
    def try_int(val):
        try:
            return (0, int(float(val)))
        except (ValueError, TypeError):
            return (1, str(val))
    all_years = sorted(all_years, key=try_int)
        
    all_series = []
    for df in all_dfs:
        for col in df.columns:
            if col not in all_series:
                all_series.append(col)

    # Initialize merged_df with correct types if possible
    merged_df = pd.DataFrame(index=all_years, columns=all_series, dtype=float)
    merged_df.index.name = "YEAR"

    for df in all_dfs:
        # Ensure df index matches merged_df index type for alignment
        new_index = []
        for y in df.index:
            try:
                new_index.append(int(float(y)))
            except (ValueError, TypeError):
                new_index.append(str(y))
        df.index = new_index

        for series in df.columns:
            series_data = df[series].dropna()
            for year, value in series_data.items():
                merged_df.at[year, series] = value

    # Report any files that couldn't be read
    if error_files:
        print(
            f"Warning: Could not read the following files: {', '.join(error_files)}"
        )

    return merged_df
