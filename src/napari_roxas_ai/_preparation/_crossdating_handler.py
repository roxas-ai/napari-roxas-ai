"""
Handles the selection and processing of crossdating files.
"""

import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from napari_roxas_ai._reader._crossdating_reader import read_crossdating_file


def is_crossdating_file(file_path: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Determine if a file is a valid crossdating candidate file.

    Only files with extensions .rwl, .tuc, or .txt are eligible.
    Files belonging to ROXAS AI measurement outputs (such as cells_table
    or rings_table) are excluded.

    Parameters
    ----------
    file_path : str
        Path of the candidate file.
    allowed_extensions : Optional[List[str]]
        Allowed extensions for crossdating files (defaults to [".rwl", ".tuc", ".txt"]).

    Returns
    -------
    bool
        True if the file is a valid crossdating input file.
    """
    if allowed_extensions is None:
        allowed_extensions = [".rwl", ".tuc", ".txt"]

    file_path_obj = Path(file_path)
    file_name = file_path_obj.name.lower()
    file_suffix = file_path_obj.suffix.lower()

    # Check extension
    if file_suffix not in [ext.lower() for ext in allowed_extensions]:
        return False

    # Exclude ROXAS AI table outputs
    if "cells_table" in file_name or "rings_table" in file_name:
        return False

    return True


def is_crossdating_output_file(file_path: str) -> bool:
    """
    Determine if a file is an existing crossdating output file (already scaled in micrometers).

    Parameters
    ----------
    file_path : str
        Path of the file.

    Returns
    -------
    bool
        True if the file is a crossdating output file.
    """
    file_name = Path(file_path).name.lower()
    return "crossdating" in file_name


DEFAULT_CROSSDATING_TEXT_EXTENSIONS = [".rwl", ".tuc", ".txt"]


class CrossdatingSelectionDialog(QDialog):
    """
    Dialog for selecting crossdating files to process and their scaling.
    """

    def __init__(
        self,
        project_directory: str,
        text_file_extensions: Optional[List[str]] = None,
        project_file_path: str = "",
        crossdating_file_extension: str = ".crossdating.txt",
        parent=None,
    ):
        """
        Initialize the crossdating selection dialog.

        Parameters
        ----------
        project_directory : str
            The project directory containing crossdating files
        text_file_extensions : List[str], optional
            List of file extensions to consider as crossdating text files (defaults to .rwl, .tuc, .txt)
        project_file_path : str
            Path to the project crossdating file (to exclude from selection)
        crossdating_file_extension : str
            The file extension for crossdating files (defaults to .crossdating.txt)
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.project_directory = project_directory
        self.crossdating_file_extension = crossdating_file_extension
        if text_file_extensions is None:
            self.text_file_extensions = DEFAULT_CROSSDATING_TEXT_EXTENSIONS
        else:
            # Filter allowed extensions to only .rwl, .tuc, .txt
            allowed_exts = [
                ext
                for ext in text_file_extensions
                if ext.lower() in DEFAULT_CROSSDATING_TEXT_EXTENSIONS
            ]
            self.text_file_extensions = (
                allowed_exts if allowed_exts else DEFAULT_CROSSDATING_TEXT_EXTENSIONS
            )
        self.project_file_path = project_file_path
        self.selected_files = []
        self.selected_scaling = 10.0  # Default to 1/100 mm (10 um)
        self.selected_prefix = "rings_series"

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

        # Output filename selection
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel("Output file name:")
        self.prefix_input = QLineEdit("rings_series")
        self.prefix_input.setAlignment(Qt.AlignRight)
        self.prefix_input.setToolTip(
            'Add a project-specific prefix, otherwise "rings_series" will be used'
        )
        self.extension_label = QLabel(self.crossdating_file_extension)
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_input)
        prefix_layout.addWidget(self.extension_label)
        layout.addLayout(prefix_layout)

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

        # Find all crossdating files in the project directory (including subdirectories)
        for ext in self.text_file_extensions:
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                found_files = glob.glob(
                    str(Path(self.project_directory) / "**" / pattern),
                    recursive=True,
                )
                for f in found_files:
                    if is_crossdating_file(f, self.text_file_extensions):
                        text_files.add(str(Path(f).resolve()))

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
        """Handle OK button click - collect selected files and settings."""
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

        # Get prefix
        raw_prefix = self.prefix_input.text().strip()
        if not raw_prefix:
            raw_prefix = "rings_series"
        # If user accidentally included extension in the prefix input, strip it
        if raw_prefix.endswith(self.crossdating_file_extension):
            raw_prefix = raw_prefix[:-len(self.crossdating_file_extension)]
        elif raw_prefix.endswith(".txt") or raw_prefix.endswith(".rwl") or raw_prefix.endswith(".tuc"):
            raw_prefix = Path(raw_prefix).stem

        self.selected_prefix = raw_prefix if raw_prefix else "rings_series"

        self.accept()

    def get_selected_files(self) -> List[str]:
        """Return the list of selected file paths."""
        return self.selected_files

    def get_selected_scaling(self) -> float:
        """Return the selected scaling factor to micrometers."""
        return self.selected_scaling

    def get_selected_prefix(self) -> str:
        """Return the selected output file prefix."""
        return self.selected_prefix

    def get_output_filename(self) -> str:
        """Return the full output filename with extension."""
        return f"{self.selected_prefix}{self.crossdating_file_extension}"


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
    # Show dialog to select crossdating files
    dialog = CrossdatingSelectionDialog(
        project_directory=project_directory,
        text_file_extensions=text_file_extensions,
        crossdating_file_extension=crossdating_file_extension,
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
    output_filename = dialog.get_output_filename()
    crossdating_file_path = Path(project_directory) / output_filename

    # Process selected files and merge with project file
    merged_df = merge_crossdating_files(
        selected_files, str(crossdating_file_path), scaling_factor
    )

    # Save merged data back to the project file
    if merged_df is not None and not merged_df.empty:
        # Use tab separator for saving the file
        merged_df.to_csv(str(crossdating_file_path), sep="\t", index=True)
        QMessageBox.information(
            None,
            "Crossdating Files Processed",
            f"Crossdating data has been processed and saved to:\n{crossdating_file_path}",
        )

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
    Read and merge multiple crossdating files to replace or update the existing target.

    Existing crossdating output files (e.g., *.crossdating.txt) are assumed to be
    already scaled in micrometers (scaling factor 1.0). Raw input files (.rwl, .tuc,
    standard .txt) are scaled using the provided `scaling_factor`.

    When merging, base output files are processed first and raw input files second,
    ensuring that raw input files gain priority in case of duplicate (series, year)
    entries.

    Parameters
    ----------
    source_files : List[str]
        List of source crossdating files to process and merge
    target_file : str
        Target crossdating file (will be replaced by new data)
    scaling_factor : float
        Scaling factor to apply to raw source files to convert them to micrometers.
        Defaults to 1.0 (no scaling).

    Returns
    -------
    Optional[pd.DataFrame]
        The merged DataFrame if successful, None otherwise
    """
    # Partition files into existing crossdating output files (base) and raw input files
    base_files = [f for f in source_files if is_crossdating_output_file(f)]
    raw_files = [f for f in source_files if not is_crossdating_output_file(f)]

    dfs_to_scale = []
    error_files = []

    # Helper function to read and conditionally scale a file
    def _read_and_scale(file_path: str, scale: float):
        success, source_df = _try_read_dataframe(file_path)
        if success and source_df is not None:
            if scale != 1.0:
                for col in source_df.columns:
                    source_df[col] = pd.to_numeric(source_df[col], errors="coerce")

                numeric_cols = source_df.select_dtypes(include=[np.number]).columns
                series_cols = [
                    col
                    for col in numeric_cols
                    if "year" not in str(col).lower() and "index" not in str(col).lower()
                ]
                source_df[series_cols] = source_df[series_cols] * scale

            dfs_to_scale.append(source_df)
        else:
            error_files.append(Path(file_path).name)
            print(f"Error reading file {file_path}")

    # Process base output files first with scale 1.0 (already in micrometers)
    for file_path in base_files:
        _read_and_scale(file_path, scale=1.0)

    # Process raw input files second with scaling_factor (gain priority on merge)
    for file_path in raw_files:
        _read_and_scale(file_path, scale=scaling_factor)

    if not dfs_to_scale:
        return None

    all_dfs = dfs_to_scale

    # Collect all years and series names from selected files
    all_years_raw = set().union(*[df.index for df in all_dfs])
    all_years = []
    for y in all_years_raw:
        if pd.isna(y):
            continue
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
