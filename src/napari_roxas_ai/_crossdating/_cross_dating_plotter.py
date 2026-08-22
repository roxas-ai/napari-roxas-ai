from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import napari.layers
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QSizePolicy
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
    RangeSlider,
    Slider,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from napari.utils.notifications import show_info
from PIL import Image
from qtpy.QtWidgets import QVBoxLayout, QWidget
from superqt import QRangeSlider

from napari_roxas_ai._edition import update_rings_geometries
from napari_roxas_ai._reader._crossdating_reader import read_crossdating_file
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._utils._callback_manager import (
    register_layer_callback,
    unregister_layer_callback,
)
from napari_roxas_ai._utils._metadata_keys import NO_REFERENCE_SERIES
from napari_roxas_ai._writer._writer import update_metadata_file

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

settings = SettingsManager()


def simplify_string(string: str) -> str:
    """
    Simplify a string by removing special characters and lowercasing it.

    Parameters
    ----------
    string : str
        The input string to simplify.

    Returns
    -------
    str
        The simplified string.
    """
    # Remove special characters and convert to lowercase
    return "".join(
        character for character in string if character.isalnum()
    ).lower()


class MatplotlibCanvas(Container):
    """A Container widget that wraps a matplotlib canvas."""

    def __init__(self, figsize=(6, 4), dpi=100):
        # Set dark theme for the figure
        self.figure = Figure(figsize=figsize, dpi=dpi, facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Set dark theme for the axes
        self.ax.set_facecolor('black')
        self.ax.tick_params(axis='both', colors='lightgrey')
        self.ax.xaxis.label.set_color('lightgrey')
        self.ax.yaxis.label.set_color('lightgrey')
        
        for spine in self.ax.spines.values():
            spine.set_edgecolor('lightgrey')

        # Create a Qt widget to hold the canvas
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        widget.setLayout(layout)
        widget.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        # Initialize the Container with our widget
        super().__init__(widgets=[])
        self.native.layout().addWidget(widget)

    def clear(self):
        """Clear the plot."""
        self.ax.clear()
        self.canvas.draw()


class CrossDatingPlotterWidget(Container):
    @property
    def _input_layer(self) -> Optional["napari.layers.Labels"]:
        """Get the single valid ring layer currently in the viewer."""
        valid_layers = self._get_valid_layers()
        return valid_layers[0] if valid_layers else None

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # bunch of attributes
        self.crossdating_files = []
        self.crossdating_columns = []
        self.plot_df = None
        self._layer_callback = None
        # Guards the reference series write while the widget itself establishes
        # the selection, so that reset_choices() does not persist a transient
        # None as "NA" before the real value is assigned.
        self._suppress_reference_store = False

        self._auto_offset_button = PushButton(
            text="Find best overlap",
            tooltip="Automatically align Reference and Sample",
        )
        self._auto_offset_button.changed.connect(self._auto_align_sample)

        # Make a combobox to choose the crossdating file path
        self._crossdating_file_combo = ComboBox(
            label="Crossdating File",
            choices=lambda widget: self.crossdating_files,
        )
        self._crossdating_file_combo.changed.connect(
            self._on_new_crossdating_file
        )

        # Create a selection widget for the column name
        self._crossdating_column_combo = ComboBox(
            label="Reference Series",
            choices=lambda widget: self.crossdating_columns,
        )
        self._crossdating_column_combo.changed.connect(
            self._on_new_crossdating_column
        )

        # Range slider for x-axis limits
        self._x_range_slider = RangeSlider(
            label="Year Range",
            min=0,
            max=100,
            step=1,
            value=(0, 100),
        )
        self._x_range_slider.changed.connect(self._on_x_range_changed)
        self._x_range_slider_was_set = False

        # Range slider for y-axis limits
        self._y_range_slider = RangeSlider(
            label="Width Range",
            min=0,
            max=100,
            step=1,
            value=(0, 100),
        )
        self._y_range_slider.changed.connect(self._on_y_range_changed)
        self._y_range_slider_was_set = False

        # Slider for offset
        self._offset_slider = Slider(
            label="Offset",
            min=-50,
            max=50,
            step=1,
            value=0,
        )
        self._offset_slider.changed.connect(
            lambda: self._plot_crossdating_data()
        )  # Update only the plot when the offset changes
        self._apply_changes_button = PushButton(
            text="Apply Changes",
            tooltip="Apply all current changes (offset, alignment) to the rings layer",
        )
        self._apply_changes_button.changed.connect(self._apply_offset_to_layer)

        # Create matplotlib canvas widget
        self.plot_widget = MatplotlibCanvas(figsize=(6, 4), dpi=100)

        # Export plot button at bottom-right
        self._export_plot_button = PushButton(
            text="Export plot",
            tooltip="Save current crossdating plot as an image in the project directory",
        )
        self._export_plot_button.changed.connect(self._export_plot)
        self._plot_footer = Container(layout="horizontal")
        self._plot_footer.append(Container())  # spacer
        self._plot_footer.append(self._export_plot_button)

        # style sliders
        self._style_rangeslider(self._x_range_slider)
        self._style_rangeslider(self._y_range_slider)

        # Append the widgets to the container
        self.extend(
            [
                self._crossdating_file_combo,
                self._crossdating_column_combo,
                self._x_range_slider,
                self._y_range_slider,
                self._offset_slider,
                self._apply_changes_button,
                self._auto_offset_button,
                self.plot_widget,
                self._plot_footer,
            ]
        )

        self._alignment_candidates = []
        self._current_alignment_data = None
        self._base_offset = 0  # Offset accumulated by auto-alignment or candidate selection
        self._roxas_visibility_threshold = 0.90  # Re-center if less than this fraction is visible
        self._alignment_buttons_container = Container()
        self._alignment_buttons_container.native.setSizePolicy(
            self._alignment_buttons_container.native.sizePolicy().Expanding,
            self._alignment_buttons_container.native.sizePolicy().Fixed,
        )

        self.insert(
            self.index(self._auto_offset_button) + 1,
            self._alignment_buttons_container
        )

        # Connect to viewer events to track layer changes
        self._viewer.layers.events.inserted.connect(self._on_layer_change)
        self._viewer.layers.events.removed.connect(self._on_layer_change)

        self._on_new_input_layer()
        self._on_new_crossdating_file()

    def _export_plot(self):
        layer = self._input_layer
        if layer is None:
            show_info("Export plot failed: no rings layer selected")
            return

        # Project directory is the anchor for relative stems
        proj = settings.get("project_directory")
        project_dir = Path(proj).resolve() if isinstance(proj, str) and proj else None

        # Prefer: project_dir / sample_stem_path.parent
        out_dir = None
        stem = layer.metadata.get("sample_stem_path")

        if project_dir is not None and isinstance(stem, str) and stem.strip():
            stem_path = Path(stem)
            # sample_stem_path is expected to be relative (e.g. "02_5/MEN.FICU_RAL16A_02_5")
            # but handle absolute defensively
            if stem_path.is_absolute():
                out_dir = stem_path.parent
            else:
                out_dir = (project_dir / stem_path).parent

        # Fallbacks
        if out_dir is None:
            layer_file = layer.metadata.get("file_path") or layer.metadata.get("path")
            if isinstance(layer_file, str) and layer_file:
                out_dir = Path(layer_file).resolve().parent
            elif project_dir is not None:
                out_dir = project_dir
            else:
                out_dir = Path.cwd()

        sample_name = layer.metadata.get("sample_name") or layer.name

        out_path = self.save_crossdating_plot_image(out_dir, sample_name)
        if out_path is None:
            show_info("Export plot failed: plot not ready")
            return

        show_info(f"Plot exported to: {out_path}")

    def _style_rangeslider(self, slider: RangeSlider) -> None:
        # Style magicgui RangeSlider handles to look like thin vertical bars instead of fat circles.
        native = slider.native

        qrange = native.findChild(QRangeSlider)
        if qrange is None:
            return

        qrange.setStyleSheet("""
        QRangeSlider::groove {
            height: 6px;
            background: #444;
            border-radius: 3px;
        }

        QRangeSlider::sub-page {
            background: #777;
            border-radius: 3px;
        }

        QRangeSlider::add-page {
            background: #333;
            border-radius: 3px;
        }

        /* THIS is the important part */
        QRangeSlider::handle {
            background: #e0e0e0;
            border: 1px solid #555;
            width: 3px;
            margin: -6px 0px;    /* makes handle taller */
            border-radius: 0px;  /* square = looks like | */
        }

        QRangeSlider::handle:hover {
            width: 6px;
            background: #ffffff;
        }
        """)

    def _get_valid_layers(self, widget=None):
        """Get layers that are both Labels type and match the rings file extension."""
        rings_extension = settings.get("file_extensions.rings_file_extension")[
            0
        ]
        valid_layers = []

        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name.endswith(
                rings_extension
            ):
                valid_layers.append(layer)

        return valid_layers

    def _on_new_input_layer(self):
        # Skip if no layer is selected
        if self._input_layer is None:
            # Clean up any previous layer callback
            self._disconnect_layer_callback()
            return

        # Connect to this layer's events
        self._connect_layer_callback()

        # Reset the range sliders for the new layer
        self._x_range_slider_was_set = False
        self._y_range_slider_was_set = False
        self._base_offset = 0
        self._offset_slider.native.blockSignals(True)
        self._offset_slider.value = 0
        self._offset_slider.native.blockSignals(False)

        pattern = f"*{''.join(settings.get('file_extensions.crossdating_file_extension'))}"

        layer = self._input_layer
        stem = Path(layer.metadata.get("sample_stem_path", layer.name))

        base_dir = None

        # layer file path (if present)
        layer_file = layer.metadata.get("path")
        if isinstance(layer_file, str) and layer_file:
            base_dir = Path(layer_file).parent

        # configured project directory
        if base_dir is None:
            proj = settings.get("project_directory")
            if isinstance(proj, str) and proj:
                base_dir = Path(proj)

        # Resolve stem if needed
        if base_dir is not None and not stem.is_absolute():
            stem = base_dir / stem

        current_path = stem.parent if stem.parent != Path(".") else (base_dir or Path.cwd())

        # Walk up the directory tree
        while True:  # Stop at root
            # Check for matches in this directory
            self.crossdating_files = list(current_path.glob(pattern))
            if self.crossdating_files:
                break

            # Stop if we reach the root directory
            if current_path == current_path.parent:
                show_info(
                    f"No crossdating file found in the directory tree for layer {self._input_layer}."
                )
                break

            # Move up to parent directory
            current_path = current_path.parent

        # Refresh the crossdating file combo. Its choices are read from
        # self.crossdating_files, which napari only re-evaluates on a layer
        # event: opening the widget while the sample is already loaded fires no
        # such event, so without this the combo stays empty, no reference
        # series is ever selected and the plot silently stays empty.
        self._crossdating_file_combo.reset_choices()
        if (
            self._crossdating_file_combo.value is None
            and self.crossdating_files
        ):
            self._crossdating_file_combo.value = self.crossdating_files[0]

        self._update_crossdating_plot()

    def _connect_layer_callback(self):
        """Connect callback to the currently selected layer."""
        # Clean up any previous callback first
        self._disconnect_layer_callback()

        if self._input_layer is not None:
            # Connect to the layer's events using the shared callback manager
            self._layer_callback = register_layer_callback(
                self._input_layer, self, self._on_layer_data_change
            )

    def _disconnect_layer_callback(self):
        """Disconnect callback from the previously selected layer."""
        if (
                self._input_layer is not None
                and self._layer_callback is not None
        ):
            unregister_layer_callback(self._input_layer, self)
            self._layer_callback = None

    def _on_layer_change(self, event=None):
        """Called when layers are added/removed in the viewer."""
        # Get valid layers
        valid_layers = self._get_valid_layers()

        # Update layer callback if needed
        # We always want to be connected to the single valid layer if it exists
        current_layer = self._input_layer
        if current_layer:
             # This will disconnect old and connect to new if it changed
             self._on_new_input_layer()
        else:
            # If no valid layers, disconnect any callbacks and clear the plot
            self._disconnect_layer_callback()
            if (
                    hasattr(self, "plot_widget")
                    and self.plot_widget is not None
            ):
                self.plot_widget.ax.clear()
                self.plot_widget.canvas.draw()
            # Clear other dependent widgets
            self.crossdating_files = []
            self.crossdating_columns = []
            self._crossdating_file_combo.reset_choices()
            self._crossdating_column_combo.reset_choices()

    def _on_layer_data_change(self, event=None):
        """Called when the data in the selected layer changes."""
        # Only respond to data, metadata, or features changes
        if event is not None:
            if event.type not in ["data", "metadata", "features"]:
                return

        if self._crossdating_column_combo.value is not None:
            # If the metadata (e.g. outmost year) changed, we want to force re-centering
            if event is not None and event.type == "metadata":
                self._x_range_slider_was_set = False
                self._y_range_slider_was_set = False

            # Update the plot if we have a valid column selected
            self._update_crossdating_plot()

    def _on_new_crossdating_file(self):
        # Skip if no crossdating file is selected
        if self._crossdating_file_combo.value is None:
            return

        # Reset the range sliders for the new crossdating file
        self._x_range_slider_was_set = False
        self._y_range_slider_was_set = False
        self._base_offset = 0
        self._offset_slider.native.blockSignals(True)
        self._offset_slider.value = 0
        self._offset_slider.native.blockSignals(False)

        # Read the crossdating file
        self.crossdating_dataframe = read_crossdating_file(
            self._crossdating_file_combo.value
        )

        # average of the crossdating file columns
        self.crossdating_dataframe["average"] = (
            self.crossdating_dataframe.mean(axis=1, skipna=True)
        )

        # Then, we order the column names by decreasing size
        column_names = sorted(
            self.crossdating_dataframe.drop("average", axis=1).columns,
            key=lambda x: len(x),
            reverse=True,
        )

        # We match the column names with the layer name
        matching_columns = [
            name
            for name in column_names
            if simplify_string(name)
               in simplify_string(self._input_layer.name)
        ]

        # We provide the matching columns first, and then remaining columns
        self.crossdating_columns = matching_columns + [
            name for name in column_names if name not in matching_columns
        ]

        # Reset choices for column combo, then select in order of priority:
        # a previously stored reference series, else the first name-matching
        # column, else the first available one.
        stored = self._input_layer.metadata.get("reference_series")
        stored_is_usable = (
            isinstance(stored, str)
            and stored != NO_REFERENCE_SERIES
            and stored in self.crossdating_columns
        )

        self._suppress_reference_store = True
        try:
            self._crossdating_column_combo.reset_choices()
            if stored_is_usable:
                self._crossdating_column_combo.value = stored
            elif matching_columns:
                self._crossdating_column_combo.value = matching_columns[0]
            elif self.crossdating_columns:
                self._crossdating_column_combo.value = (
                    self.crossdating_columns[0]
                )
        finally:
            self._suppress_reference_store = False

        # Persist whatever ended up selected, including "NA" when nothing did
        self._store_reference_series()

        if not self.crossdating_columns:
            show_info(
                f"No data found in the crossdating file {self._crossdating_file_combo.value} for layer {self._input_layer}."
            )

        self._update_crossdating_plot()

    def _sample_metadata_path(self) -> Optional[Path]:
        """
        Path of the metadata file belonging to the current input layer.

        Mirrors the resolution used by the writer: the sample stem is stored
        relative to the project directory.
        """
        layer = self._input_layer
        if layer is None:
            return None

        metadata_file_extension = "".join(
            settings.get("file_extensions.metadata_file_extension")
        )

        stem = layer.metadata.get("sample_stem_path")
        if isinstance(stem, str) and stem.strip():
            proj = settings.get("project_directory")
            if isinstance(proj, str) and proj:
                return Path(
                    f"{(Path(proj).resolve() / stem)}{metadata_file_extension}"
                )

        # Fallback: next to the layer's own file
        layer_file = layer.metadata.get("file_path")
        sample_name = layer.metadata.get("sample_name")
        if isinstance(layer_file, str) and layer_file and sample_name:
            return (
                Path(layer_file).parent
                / f"{sample_name}{metadata_file_extension}"
            )

        return None

    def _store_reference_series(self) -> None:
        """
        Persist the selected reference series for the current sample.

        Written straight to the metadata file rather than waiting for a layer
        save, so that the selection survives even if nothing else is saved.
        Only this one key is passed, so no other metadata can be touched.
        """
        if self._suppress_reference_store:
            return

        layer = self._input_layer
        if layer is None:
            return

        value = self._crossdating_column_combo.value or NO_REFERENCE_SERIES
        if layer.metadata.get("reference_series") == value:
            return  # nothing changed, skip the file write

        layer.metadata["reference_series"] = value

        path = self._sample_metadata_path()
        if path is None or not path.exists():
            return

        try:
            update_metadata_file(
                str(path),
                {
                    "sample_name": layer.metadata.get("sample_name"),
                    "reference_series": value,
                },
                ("reference_series",),
            )
        except OSError as e:
            print(f"[crossdating] Could not store reference_series: {e}")

    def _on_new_crossdating_column(self):
        """Called when a new reference series is selected."""
        # Store first, so that the selection is persisted even when the value is
        # None (no usable column) and the early return below applies.
        self._store_reference_series()

        if self._crossdating_column_combo.value is None:
            return

        # Reset the range sliders for the new reference series
        # Note: we keep the year range (x-axis) persistent as requested by the user,
        # but the y-axis (width) should autoscale to the new series' range.
        self._y_range_slider_was_set = False

        # Clear any existing alignment candidates as they were for the previous reference
        self._clear_alignment_buttons()
        self._alignment_candidates = []
        self._current_alignment_data = None

        # Update the plot
        self._update_crossdating_plot()

    def _update_crossdating_plot(self):
        # Skip if no crossdating column is selected
        if self._crossdating_column_combo.value is None:
            if hasattr(self, "plot_widget") and self.plot_widget is not None:
                self.plot_widget.ax.clear()
                self.plot_widget.canvas.draw()
            return

        layer = self._input_layer
        if layer is None:
            return

        # Ensure features are present
        feats = getattr(layer, "features", None)
        if feats is None or feats.empty or "YEAR" not in feats.columns:
            # If no features yet, clear plot and return
            if hasattr(self, "plot_widget") and self.plot_widget is not None:
                self.plot_widget.ax.clear()
                self.plot_widget.canvas.draw()
            return

        # Get reference series
        try:
            reference_series = self.crossdating_dataframe[
                self._crossdating_column_combo.value
            ]
        except (KeyError, AttributeError):
            show_info(f"Reference series '{self._crossdating_column_combo.value}' not found in data.")
            return

        # Get the average series
        average_series = self.crossdating_dataframe.get("average", pd.Series(dtype=float))

        # Get the layer rings series
        layer_df = layer.features.set_index("YEAR").copy()

        # Compute the difference with previous year
        if "cells_above" in layer_df.columns:
            # Sort by year to ensure correct diff
            layer_df = layer_df.sort_index()

            # The first value in cells_above is the area above the first ring.
            # The ring width for year Y is cells_above(Y) - cells_above(Y-1).
            # For the first ring in the table, it doesn't have a predecessor in the table.
            # However, the table usually contains all rings.

            # If we want to mirror the previous logic:
            vals = layer_df["cells_above"].values
            diffs = np.diff(vals, prepend=vals[0])  # prepend to keep same length
            layer_df["ring_width"] = diffs

            # Note: the previous logic did:
            # layer_df.iloc[1:, layer_df.columns.tolist().index("cells_above")] = np.diff(layer_df["cells_above"].values)
            # which modified cells_above in place and left the first one as is (which is area, not width).
            # This seems slightly inconsistent but let's stick to a cleaner version if possible,
            # or keep it if it's what's expected.

            # Actually, the previous logic was:
            # layer_df.iloc[1:, index] = np.diff(...)
            # This means layer_df["cells_above"].iloc[0] remained the TOTAL area above the first ring.
            # Subsequent ones became widths.

            # Let's keep it exactly as it was but more robustly:
            idx = layer_df.columns.get_loc("cells_above")
            layer_df.iloc[1:, idx] = np.diff(layer_df["cells_above"].values)
        else:
            show_info("Layer features missing 'cells_above' column.")
            return

        # Removed values of disabled years
        if "enabled" in layer_df.columns:
            layer_df.loc[~layer_df["enabled"], "cells_above"] = np.nan

        # Create ring width series
        # Ensure data and metadata are present
        if not hasattr(layer, "data") or "spatial_resolution" not in layer.metadata:
            return

        width_series = layer_df["cells_above"] / (
                layer.data.shape[1]
                * layer.metadata["spatial_resolution"]
        )

        # Clear plot_df and rebuild it to ensure no stale data
        self.plot_df = pd.DataFrame(index=self.crossdating_dataframe.index)
        self.plot_df["reference_series"] = reference_series
        self.plot_df["average"] = average_series

        # Merge layer_series (width_series) - it might have different years
        self.plot_df = self.plot_df.join(width_series.rename("layer_series"), how="outer")
        self.plot_df.index.name = "YEAR"

        # Update the plot
        self._plot_crossdating_data()

    def _plot_crossdating_data(self, target_range: Optional[tuple[int, int]] = None):
        """Plot the crossdating data comparison"""
        if self.plot_df is None or self.plot_df.empty:
            return

        # Calculate correlation and overlapping period
        # The ROXAS series is shifted by the total offset
        total_offset = self._base_offset + self._offset_slider.value

        # Create a shifted version of layer_series for correlation calculation and scaling
        # This aligns the ROXAS data with the reference data at the new (offset) position
        shifted_layer_series = self.plot_df["layer_series"].copy()
        shifted_layer_series.index = shifted_layer_series.index + total_offset

        # Calculate correlation and overlapping period for labels
        corr_df = pd.concat([
            self.plot_df["reference_series"],
            shifted_layer_series,
            self.plot_df["average"]
        ], axis=1)
        corr_df.columns = ["reference", "roxas", "average"]

        mask_ref = corr_df["reference"].notna() & corr_df["roxas"].notna()
        overlap_years_ref = corr_df.index[mask_ref].tolist()

        if overlap_years_ref:
            overlap_min_ref = min(overlap_years_ref)
            overlap_max_ref = max(overlap_years_ref)
            r_ref = corr_df["reference"].corr(corr_df["roxas"])
            glk_ref = self.calculate_glk(
                corr_df.loc[overlap_years_ref, "reference"].to_numpy(),
                corr_df.loc[overlap_years_ref, "roxas"].to_numpy()
            )
            period_ref = f"{overlap_min_ref}-{overlap_max_ref}"
        else:
            r_ref = np.nan
            glk_ref = np.nan
            period_ref = "no overlap"

        # Calculate for Average
        mask_avg = corr_df["average"].notna() & corr_df["roxas"].notna()
        overlap_years_avg = corr_df.index[mask_avg].tolist()
        if overlap_years_avg:
            r_avg = corr_df["average"].corr(corr_df["roxas"])
            glk_avg = self.calculate_glk(
                corr_df.loc[overlap_years_avg, "average"].to_numpy(),
                corr_df.loc[overlap_years_avg, "roxas"].to_numpy()
            )
        else:
            r_avg = np.nan
            glk_avg = np.nan

        # Clear the previous plot
        self.plot_widget.ax.clear()

        # Build labels
        layer = self._input_layer
        image_id = layer.metadata.get("sample_name") or layer.name
        roxas_label = f"RXS: {image_id} ({period_ref})"

        ref_name = str(self._crossdating_column_combo.value)
        glk_ref_text = f", glk={glk_ref:.0f}" if not np.isnan(glk_ref) else ""
        r_ref_text = f": r={r_ref:.3f}{glk_ref_text}" if not np.isnan(r_ref) else ""
        ref_label = f"{ref_name}{r_ref_text}"
        
        glk_avg_text = f", glk={glk_avg:.0f}" if not np.isnan(glk_avg) else ""
        r_avg_text = f": r={r_avg:.3f}{glk_avg_text}" if not np.isnan(r_avg) else ""
        avg_label = f"Average{r_avg_text}"

        # Plot both series
        years = self.plot_df.index.to_numpy(dtype=int)

        # Plot the ROXAS series first (top layer in legend)
        self.plot_widget.ax.plot(
            years + total_offset,
            self.plot_df["layer_series"],
            color='red',
            linestyle='-',
            label=roxas_label,
            zorder=3,
        )

        # Plot the reference second (middle layer in legend)
        self.plot_widget.ax.plot(
            years,
            self.plot_df["reference_series"],
            color='yellow',
            linestyle='-',
            label=ref_label,
            zorder=2,
        )

        # Plot the average last (bottom layer in legend)
        self.plot_widget.ax.plot(
            years,
            self.plot_df["average"],
            color='white',
            linestyle='-',
            label=avg_label,
            zorder=1,
        )

        # Set labels and title
        #self.plot_widget.ax.set_xlabel("Year")
        self.plot_widget.ax.set_ylabel("Width (\u03BCm)", color='lightgrey')

        legend = self.plot_widget.ax.legend(
            loc="best",
            facecolor='black',
            edgecolor='lightgrey'
        )
        
        # Color each legend text according to its series
        # We assume the order: ROXAS, Reference, Average
        series_colors = ['red', 'yellow', 'white']
        for text, color in zip(legend.get_texts(), series_colors):
            text.set_color(color)

        # Configure grid: 5-year vertical lines (minor), 10-year labels (major)
        self.plot_widget.ax.xaxis.set_major_locator(MultipleLocator(10))
        self.plot_widget.ax.xaxis.set_minor_locator(MultipleLocator(5))
        
        # Major vertical grid lines (every 10 years) - solid
        self.plot_widget.ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.5, color='lightgrey')
        # Minor vertical grid lines (every 5 years) - dashed
        self.plot_widget.ax.grid(True, which='minor', axis='x', linestyle='--', alpha=0.3, color='lightgrey')
        # Horizontal main grid lines - solid
        self.plot_widget.ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.3, color='lightgrey')

        # Get the min and max years for x-axis
        # We need to account for both the reference series and the shifted ROXAS series
        total_offset = self._base_offset + self._offset_slider.value
        roxas_years_raw = self.plot_df["layer_series"].dropna().index.to_numpy()
        roxas_years = roxas_years_raw + total_offset

        if len(roxas_years) > 0:
            min_year = int(min(min(years), min(roxas_years)) - 10)
            max_year = int(max(max(years), max(roxas_years)) + 10)
        else:
            min_year = int(min(years) - 10)
            max_year = int(max(years) + 10)

        # Update x slider range but preserve values if possible
        self._x_range_slider.native.blockSignals(True)
        self._x_range_slider.min = min_year
        self._x_range_slider.max = max_year
        self._x_range_slider.native.blockSignals(False)

        # If the value of the slider has been initialized
        if self._x_range_slider_was_set:

            # Get current x slider values
            current_x_low, current_x_high = self._x_range_slider.value

            # Check if enough of the ROXAS series is still visible
            # If it isn't, we should re-center
            roxas_visible = False
            if len(roxas_years) > 0:
                roxas_min = min(roxas_years)
                roxas_max = max(roxas_years)
                roxas_width = roxas_max - roxas_min

                if roxas_width > 0:
                    # Calculate intersection of [roxas_min, roxas_max] and [current_x_low, current_x_high]
                    visible_min = max(roxas_min, current_x_low)
                    visible_max = min(roxas_max, current_x_high)
                    visible_width = max(0, visible_max - visible_min)

                    # Trigger re-centering if visible width is less than threshold
                    if visible_width >= (self._roxas_visibility_threshold * roxas_width):
                        roxas_visible = True
                else:
                    # Single point curve is visible if within range
                    if current_x_low <= roxas_min <= current_x_high:
                        roxas_visible = True

            # Compute new x view range that preserves as much of previous view as possible
            if roxas_visible:
                new_x_low = (
                    current_x_low
                    if (current_x_low >= min_year) and (current_x_low < max_year)
                    else (min(roxas_years) - 10 if len(roxas_years) > 0 else min_year)
                )
                new_x_high = (
                    current_x_high
                    if (current_x_high <= max_year) and (current_x_high > min_year)
                    else (max(roxas_years) + 10 if len(roxas_years) > 0 else max_year)
                )
            else:
                # Force re-center if ROXAS is not visible in current view
                new_x_low = min(roxas_years) - 10 if len(roxas_years) > 0 else min_year
                new_x_high = max(roxas_years) + 10 if len(roxas_years) > 0 else max_year

            # Only update if the previous x range isn't valid anymore
            if new_x_low != current_x_low or new_x_high != current_x_high:
                self._x_range_slider.native.blockSignals(True)
                self._x_range_slider.value = (new_x_low, new_x_high)
                self._x_range_slider.native.blockSignals(False)
        # Otherwise, we don't want to consider the current values as they are defaults with no meaning
        else:
            # We want to center the initial view on the ROXAS series if possible
            if len(roxas_years) > 0:
                self._x_range_slider.native.blockSignals(True)
                self._x_range_slider.value = (min(roxas_years) - 10, max(roxas_years) + 10)
                self._x_range_slider.native.blockSignals(False)
                self._x_range_slider_was_set = True
            else:
                self._x_range_slider.native.blockSignals(True)
                self._x_range_slider.value = (min_year, max_year)
                self._x_range_slider.native.blockSignals(False)
                # We don't mark it as set yet, because roxas_years were empty.
                # It should try again once data is available.

        # Set the x axis limits before calculating y limits
        self.plot_widget.ax.set_xlim(self._x_range_slider.value)

        # UPDATED: We use the target_range (if provided) or the slider's value to determine visible range.
        # This is crucial for first-time auto-alignment where the slider isn't yet visually updated.
        if target_range is not None:
            x_min, x_max = target_range
        else:
            x_min, x_max = self._x_range_slider.value

        # Calculate ROXAS years with offset
        # roxas_years was already calculated above as np.array(years) + total_offset
        # We reuse it here.

        # Filter visible values
        visible_roxas = self.plot_df["layer_series"].copy()
        visible_roxas.index = np.array(years) + total_offset
        visible_roxas = visible_roxas[(visible_roxas.index >= x_min) & (visible_roxas.index <= x_max)].dropna()

        visible_ref = self.plot_df["reference_series"].copy()
        visible_ref = visible_ref[(visible_ref.index >= x_min) & (visible_ref.index <= x_max)].dropna()

        all_visible_series = [visible_roxas, visible_ref]

        visible_avg = self.plot_df["average"].copy()
        visible_avg = visible_avg[(visible_avg.index >= x_min) & (visible_avg.index <= x_max)].dropna()
        all_visible_series.append(visible_avg)

        all_visible_values = pd.concat(all_visible_series)

        if not all_visible_values.empty:
            min_value = float(all_visible_values.min())
            max_value = float(all_visible_values.max())
            padding = (max_value - min_value) * 0.1 if max_value > min_value else 10.0
            min_value = int(np.floor(max(0.0, min_value - padding)))
            max_value = int(np.ceil(max_value + padding))
        else:
            min_value = 0
            max_value = 100

        # Reset y slider range to new visible data bounds
        self._y_range_slider.native.blockSignals(True)
        # We ensure min is always <= max to avoid issues during update
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        
        self._y_range_slider.min = min_value
        self._y_range_slider.max = max_value
        self._y_range_slider.native.blockSignals(False)

        if not self._y_range_slider_was_set:
            self._y_range_slider.native.blockSignals(True)
            self._y_range_slider.value = (min_value, max_value)
            self._y_range_slider.native.blockSignals(False)
            # Do NOT set self._y_range_slider_was_set = True here.
            # It should only be set to True by manual user interaction.

            # Set the y axis limits
            self.plot_widget.ax.set_ylim(min_value, max_value)
        else:
            # Use manual slider values if locked
            self.plot_widget.ax.set_ylim(self._y_range_slider.value)

        # Redraw the canvas
        self.plot_widget.figure.tight_layout()
        self.plot_widget.canvas.draw()

    def _on_x_range_changed(self):
        """Called when the Year Range slider is manually adjusted."""
        if self.plot_df is None or self.plot_df.empty:
            return

        # Mark X as manually set
        self._x_range_slider_was_set = True

        # Refresh the plot (this will handle y-axis auto-scaling if not locked)
        self._plot_crossdating_data()

    def _on_y_range_changed(self):
        """Called when the Width Range slider is manually adjusted."""
        if self.plot_df is None or self.plot_df.empty:
            return

        # Mark Y as manually set
        self._y_range_slider_was_set = True

        # Directly update the plot y-axis limits
        y_min, y_max = self._y_range_slider.value
        self.plot_widget.ax.set_ylim(y_min, y_max)
        self.plot_widget.canvas.draw()

    def _apply_offset_to_layer(self):
        """Apply the offset to the current input layer."""
        if self._input_layer is None:
            return

        input_layer = self._input_layer

        # Get the current total offset value
        total_offset = self._base_offset + self._offset_slider.value
        new_last_year = (
                input_layer.metadata["rings_outmost_complete_year"] + total_offset
        )

        n = len(input_layer.features) if getattr(input_layer, "features", None) is not None else 0
        if n > 0 and "YEAR" in input_layer.features.columns:
            start = int(new_last_year) - n + 1
            input_layer.features = input_layer.features.copy()
            input_layer.features["YEAR"] = list(range(start, int(new_last_year) + 1))

        input_layer.metadata["rings_outmost_complete_year"] = new_last_year
        new_rings_table, new_rings_raster, new_colormap = (
            update_rings_geometries(
                rings_table=input_layer.features,
                last_year=new_last_year,
                image_shape=input_layer.data.shape,
            )
        )
        input_layer.data = new_rings_raster
        input_layer.features = new_rings_table
        input_layer.colormap = new_colormap

        # Re-establish x-range to +/- 10 years of the now shifted ROXAS series
        if n > 0:
            # We set the slider value BEFORE resetting the offset to ensure
            # _plot_crossdating_data (triggered by offset reset) has the final X range.
            self._x_range_slider.native.blockSignals(True)
            self._x_range_slider.value = (start - 10, int(new_last_year) + 10)
            self._x_range_slider.native.blockSignals(False)

        # Reset the base offset and the offset slider to 0 and center its range to -50, 50 (triggers _plot_crossdating_data)
        self._base_offset = 0
        self._offset_slider.min = -50
        self._offset_slider.max = 50
        self._offset_slider.native.blockSignals(True)
        self._offset_slider.value = 0
        self._offset_slider.native.blockSignals(False)

        # Ensure plot data is updated (re-calculates width_series and triggers _plot_crossdating_data)
        self._update_crossdating_plot()

        # Update the RingsLayerEditorWidget spinbox if it exists
        self._sync_rings_editor_year()

    def _sync_rings_editor_year(self):
        """Find the RingsLayerEditorWidget and update its year spinbox."""
        from napari_roxas_ai._edition._rings_layer_editor import (
            RingsLayerEditorWidget,
        )

        # Look for the widget in the viewer's window
        if hasattr(self._viewer.window, "_qt_window"):
            for dock in self._viewer.window._qt_window.findChildren(QWidget):
                # Using QWidget as a broad search, then checking class name
                # magicgui widgets are wrapped in QWidget
                if "RingsLayerEditorWidget" in str(type(dock)):
                    # If it's the right widget class, it might be the magicgui container or its native widget
                    # Let's try to find the actual container object if possible.
                    # Usually, dock widgets have the container as an attribute or it's the 'dock' itself if it was added via add_dock_widget
                    pass

            # Alternative: iterate through dock widgets
            for name, dock_widget in self._viewer.window._dock_widgets.items():
                widget = dock_widget.widget()
                if isinstance(widget, RingsLayerEditorWidget):
                    widget._update_year_spinbox()
                # Also check if it's a magicgui container wrapping it
                elif hasattr(widget, "_magic_widget") and isinstance(widget._magic_widget, RingsLayerEditorWidget):
                    widget._magic_widget._update_year_spinbox()

    @staticmethod
    def calculate_glk(ref: np.ndarray, rox: np.ndarray) -> float:
        """
        Calculate the "Gleichläufigkeit" (GLK), an index of synchronous growth changes.
        """
        k = len(ref)
        if k < 2:
            return np.nan

        # Directions of change: Ref(i) - Ref(i-1)
        # Using np.diff: diff[i] = x[i+1] - x[i]
        d_ref = np.diff(ref)
        d_rox = np.diff(rox)

        # iref = 0.5 if > 0, -0.5 if < 0, 0 if == 0
        iref = np.zeros_like(d_ref, dtype=float)
        iref[d_ref > 0] = 0.5
        iref[d_ref < 0] = -0.5

        irox = np.zeros_like(d_rox, dtype=float)
        irox[d_rox > 0] = 0.5
        irox[d_rox < 0] = -0.5

        glk = np.sum(np.abs(iref + irox))
        glk = glk / (k - 1)

        # Convert to percentage (0-100)
        return glk * 100

    def _auto_align_sample(self):
        if self.plot_df is None or self.plot_df.empty:
            show_info("Auto-align failed: no data")
            return

        self._clear_alignment_buttons()

        self._alignment_candidates = self._compute_alignment_candidates(top_k=5)

        if not self._alignment_candidates:
            show_info("No valid alignments found")
            return

        best = self._alignment_candidates[0]
        self._current_alignment_data = best
        self._apply_alignment(best)

        self._update_alignment_buttons()

    def _apply_alignment(self, candidate: dict):
        layer = self._input_layer
        if layer is None:
            return

        self._current_alignment_data = candidate

        target_end = int(candidate["end_year"])
        target_start = int(candidate["start_year"])

        # Calculate required offset relative to current layer state
        current_end = layer.metadata.get("rings_outmost_complete_year", 0)
        offset = target_end - current_end

        # Store the found offset as base_offset and reset the slider to 0
        self._base_offset = offset
        self._offset_slider.min = -50
        self._offset_slider.max = 50
        self._offset_slider.native.blockSignals(True)
        self._offset_slider.value = 0
        self._offset_slider.native.blockSignals(False)

        # Updating the sliders will trigger plot updates via their connected callbacks.
        # This is instantaneous as it avoids redundant layer rasterization and table updates.

        target_range = (target_start - 10, target_end + 10)
        self._x_range_slider.native.blockSignals(True)
        self._x_range_slider.value = target_range
        self._x_range_slider.native.blockSignals(False)

        # Explicitly trigger plot update with target_range for correct y-axis auto-scaling
        self._plot_crossdating_data(target_range=target_range)

    def _compute_alignment_candidates(self, plot_df=None, top_k: int = 4):
        if plot_df is None:
            plot_df = self.plot_df

        sample_series = (
            plot_df["layer_series"]
            .dropna()
            .sort_index()
        )
        reference_series = (
            plot_df["reference_series"]
            .dropna()
            .sort_index()
        )

        sample_vals = sample_series.to_numpy(dtype=float)
        ref_vals = reference_series.to_numpy(dtype=float)
        ref_years = reference_series.index.to_numpy()

        n_sample = len(sample_vals)
        n_ref = len(ref_vals)

        if n_sample < 5 or n_ref < n_sample:
            return []

        # Pearson correlation vectorized:
        # r = sum((x - mx) * (y - my)) / sqrt(sum((x - mx)^2) * sum((y - my)^2))
        #   = sum(x' * y) / sqrt(sum(x'^2) * sum((y - my)^2))
        # where x' = x - mx (demeaned sample)
        # my is the rolling mean of y (reference window)

        x = sample_vals
        mx = np.mean(x)
        x_prime = x - mx
        sum_x_prime_sq = np.sum(x_prime ** 2)

        if sum_x_prime_sq == 0:
            return []

        # Numerator: sum(x' * y)
        # np.correlate(ref, sample_demeaned, mode='valid') gives sum(ref[i:i+W] * x_prime)
        numerator = np.correlate(ref_vals, x_prime, mode='valid')

        # Denominator: sqrt(sum_x_prime_sq * sum((y - my)^2))
        # sum((y - my)^2) = sum(y^2) - n_sample * my^2
        # where my = sum(y) / n_sample

        # Use rolling sums for y and y^2
        cumsum_y = np.cumsum(np.insert(ref_vals, 0, 0))
        cumsum_y_sq = np.cumsum(np.insert(ref_vals ** 2, 0, 0))

        sum_y = cumsum_y[n_sample:] - cumsum_y[:-n_sample]
        sum_y_sq = cumsum_y_sq[n_sample:] - cumsum_y_sq[:-n_sample]

        my = sum_y / n_sample
        sum_y_centered_sq = sum_y_sq - n_sample * (my ** 2)

        # Handle numerical precision issues (e.g. slightly negative due to floating point)
        sum_y_centered_sq = np.maximum(sum_y_centered_sq, 0)

        denominator = np.sqrt(sum_x_prime_sq * sum_y_centered_sq)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            corrs = numerator / denominator

        # Build candidates
        candidates = []
        for i, corr in enumerate(corrs):
            if np.isnan(corr):
                continue

            # Calculate GLK for this candidate
            # ref_vals[i : i + n_sample] is the window of the reference
            ref_window = ref_vals[i: i + n_sample]
            glk = self.calculate_glk(ref_window, sample_vals)

            candidates.append({
                "start_index": i,
                "start_year": int(ref_years[i]),
                "end_year": int(ref_years[i + n_sample - 1]),
                "corr": float(corr),
                "glk": float(glk),
            })

        candidates.sort(key=lambda x: x["corr"], reverse=True)
        return candidates[:top_k]

    def _update_alignment_buttons(self):
        self._alignment_buttons_container.widgets = []

        for i, candidate in enumerate(self._alignment_candidates[1:4]):
            glk_text = f", glk={candidate['glk']:.0f}" if not np.isnan(candidate["glk"]) else ""
            btn = PushButton(
                text=f"{candidate['start_year']}–{candidate['end_year']} "
                     f"(r={candidate['corr']:.3f}{glk_text})"
            )

            def make_callback(index, button):
                def callback(_):
                    try:
                        # Check if the button still exists
                        if not hasattr(button, "native") or button.native is None:
                            return
                        
                        # Current candidate for this button
                        cand = self._alignment_candidates[index + 1]

                        # Data of the alignment we are about to replace
                        old_data = self._current_alignment_data

                        # Apply the new alignment
                        self._apply_alignment(cand)

                        # Ensure plot is updated with correct y-scaling for new alignment
                        target_range = (int(cand["start_year"]) - 10, int(cand["end_year"]) + 10)
                        self._x_range_slider.native.blockSignals(True)
                        self._x_range_slider.value = target_range
                        self._x_range_slider.native.blockSignals(False)
                        self._plot_crossdating_data(target_range=target_range)

                        # Update the button text with the replaced alignment's data
                        if old_data:
                            old_glk_text = f", glk={old_data['glk']:.0f}" if not np.isnan(old_data["glk"]) else ""
                            new_text = (
                                f"{old_data['start_year']}–{old_data['end_year']} "
                                f"(r={old_data['corr']:.3f}{old_glk_text})"
                            )
                            button.text = new_text
                            # Update the candidate stored for next click
                            self._alignment_candidates[index + 1] = old_data
                    except RuntimeError:
                        # Catch the case where the C++ object was deleted
                        pass

                return callback

            btn.changed.connect(make_callback(i, btn))

            self._alignment_buttons_container.append(btn)

    def _clear_alignment_buttons(self):
        container = self._alignment_buttons_container
        layout = container.native.layout()

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        container.widgets = []

    def save_crossdating_plot_image(self, out_dir: Union[str, Path], sample_name: str) -> Optional[Path]:
        """Save the currently displayed crossdating plot as a JPG."""
        if self.plot_df is None or self.plot_df.empty:
            return None
        if self._crossdating_column_combo.value is None:
            return None

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{sample_name}_ReferenceSeries.jpg"

        # Save the exact figure that is shown in the UI
        self.plot_widget.figure.savefig(
            out_path,
            dpi=200,
            bbox_inches="tight",
            facecolor="black",
        )
        return out_path





