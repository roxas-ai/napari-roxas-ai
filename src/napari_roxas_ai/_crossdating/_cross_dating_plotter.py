from pathlib import Path
from typing import TYPE_CHECKING

import napari.layers
import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
    RangeSlider,
    Slider,
)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from napari.utils.notifications import show_info
from PIL import Image
from qtpy.QtWidgets import QVBoxLayout, QWidget

from napari_roxas_ai._edition import update_rings_geometries
from napari_roxas_ai._reader._crossdating_reader import read_crossdating_file
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._utils._callback_manager import (
    register_layer_callback,
    unregister_layer_callback,
)

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
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Create a Qt widget to hold the canvas
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        widget.setLayout(layout)

        # Initialize the Container with our widget
        super().__init__(widgets=[])
        self.native.layout().addWidget(widget)

    def clear(self):
        """Clear the plot."""
        self.ax.clear()
        self.canvas.draw()


class CrossDatingPlotterWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # bunch of attributes
        self.crossdating_files = []
        self.crossdating_columns = []
        self.plot_df = None
        self._layer_callback = None

        # Create a layer selection widget filtered by layer type
        self._input_layer_combo = ComboBox(
            label="Rings Layer",
            annotation="napari.layers.Labels",  # This enables basic type filtering
            choices=self._get_valid_layers,  # Custom filtering function
        )
        # Get the crossdating file for the selected layer
        self._input_layer_combo.changed.connect(self._on_new_input_layer)

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
            self._update_crossdating_plot
        )

        # Checkbox to show/hide the average of the crossdating file columns
        self._plot_average_checkbox = CheckBox(
            label="Plot Columns Average",
            value=True,
        )
        self._plot_average_checkbox.changed.connect(
            self._update_crossdating_plot
        )

        # Range slider for x-axis limits
        self._x_range_slider = RangeSlider(
            label="Year Range",
            min=0,
            max=100,
            step=1,
            value=(0, 100),
        )
        self._x_range_slider.changed.connect(self._update_plot_limits)
        self._x_range_slider_was_set = False

        # Range slider for y-axis limits
        self._y_range_slider = RangeSlider(
            label="Width Range",
            min=0,
            max=100,
            step=0.01,
            value=(0, 100),
        )
        self._y_range_slider.changed.connect(self._update_plot_limits)
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
            self._update_crossdating_plot
        )  # Update the plot when the offset changes
        self._offset_apply_button = PushButton(
            text="Apply Offset",
            tooltip="Apply the offset to the current input layer",
        )
        self._offset_apply_button.changed.connect(self._apply_offset_to_layer)

        # Create matplotlib canvas widget
        self.plot_widget = MatplotlibCanvas(figsize=(6, 4), dpi=100)

        # Append the widgets to the container
        self.extend(
            [
                self._input_layer_combo,
                self._crossdating_file_combo,
                self._crossdating_column_combo,
                self._plot_average_checkbox,
                self._x_range_slider,
                self._y_range_slider,
                self._offset_slider,
                self._offset_apply_button,
                self.plot_widget,
            ]
        )

        self._on_new_input_layer()
        self._on_new_crossdating_file()

        # Connect to viewer events to track layer changes
        self._viewer.layers.events.inserted.connect(self._on_layer_change)
        self._viewer.layers.events.removed.connect(self._on_layer_change)

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
        if self._input_layer_combo.value is None:
            # Clean up any previous layer callback
            self._disconnect_layer_callback()
            return

        # Connect to this layer's events
        self._connect_layer_callback()

        pattern = f"*{''.join(settings.get('file_extensions.crossdating_file_extension'))}"

        current_path = Path(
            self._input_layer_combo.value.metadata["sample_stem_path"]
        ).parent

        # Walk up the directory tree
        while True:  # Stop at root
            # Check for matches in this directory
            self.crossdating_files = list(current_path.glob(pattern))
            if self.crossdating_files:
                break

            # Stop if we reach the root directory
            if current_path == current_path.parent:
                show_info(
                    f"No crossdating file found in the directory tree for layer {self._input_layer_combo.value}."
                )
                break

            # Move up to parent directory
            current_path = current_path.parent

        self._update_crossdating_plot()

    def _connect_layer_callback(self):
        """Connect callback to the currently selected layer."""
        # Clean up any previous callback first
        self._disconnect_layer_callback()

        if self._input_layer_combo.value is not None:
            # Connect to the layer's events using the shared callback manager
            self._layer_callback = register_layer_callback(
                self._input_layer_combo.value, self, self._on_layer_data_change
            )

    def _disconnect_layer_callback(self):
        """Disconnect callback from the previously selected layer."""
        if (
            self._input_layer_combo.value is not None
            and self._layer_callback is not None
        ):
            unregister_layer_callback(self._input_layer_combo.value, self)
            self._layer_callback = None

    def _on_layer_change(self, event=None):
        """Called when layers are added/removed in the viewer."""
        # Update the combo box choices
        self._input_layer_combo.reset_choices()

        # Get valid layers
        valid_layers = self._get_valid_layers()

        # If our current selection is no longer valid, update it
        if self._input_layer_combo.value not in valid_layers:
            if valid_layers:
                # If we have valid layers, select the first one
                self._input_layer_combo.value = valid_layers[0]
            else:
                # If no valid layers, don't try to set to None directly
                # Instead disconnect any callbacks and clear the plot
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
        if (
            event.type == "data"
            or event.type == "metadata"
            or event.type == "features"
        ) and self._crossdating_column_combo.value is not None:

            # Update the plot if we have a valid column selected
            self._update_crossdating_plot()

    def _on_new_crossdating_file(self):
        # Skip if no crossdating file is selected
        if self._crossdating_file_combo.value is None:
            return

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
            in simplify_string(self._input_layer_combo.value.name)
        ]

        # We provice the matching columns first, and then remaining columns
        self.crossdating_columns = matching_columns + [
            name for name in column_names if name not in matching_columns
        ]

        if not self.crossdating_columns:
            show_info(
                f"No data found in the crossdating file {self._crossdating_file_combo.value} for layer {self._input_layer_combo.value}."
            )

        self._update_crossdating_plot()

    def _update_crossdating_plot(self):
        # Skip if no crossdating column is selected
        if self._crossdating_column_combo.value is None:
            return

        # Get reference series
        reference_series = self.crossdating_dataframe[
            self._crossdating_column_combo.value
        ]

        # Get the average series
        average_series = self.crossdating_dataframe["average"]

        # Get the layer rings series
        layer_df = self._input_layer_combo.value.features.set_index(
            "ring_year"
        ).copy()

        # Compute the difference with previous year
        layer_df.iloc[1:, layer_df.columns.tolist().index("cells_above")] = (
            np.diff(layer_df["cells_above"].values)
        )

        # Removed values of disabled years
        layer_df.loc[~layer_df["enabled"], "cells_above"] = None

        # Create ring width series
        width_series = layer_df["cells_above"] / (
            self._input_layer_combo.value.data.shape[1]
            * self._input_layer_combo.value.metadata["sample_scale"]
        )

        self.plot_df = pd.DataFrame()
        self.plot_df = pd.concat(
            [
                self.plot_df,
                reference_series.rename("reference_series"),
                average_series.rename("average"),
                width_series.rename("layer_series"),
            ],
            axis=1,
        ).rename_axis("ring_year")

        # Update the plot
        self._plot_crossdating_data()

    def _plot_crossdating_data(self):
        """Plot the crossdating data comparison"""
        if self.plot_df is None or self.plot_df.empty:
            return

        # Clear the previous plot
        self.plot_widget.ax.clear()

        # Plot both series
        years = self.plot_df.index.astype(int).tolist()
        self.plot_widget.ax.plot(
            years,
            self.plot_df["reference_series"],
            "b-",
            label="Reference Series",
        )
        self.plot_widget.ax.plot(
            np.array(years) + self._offset_slider.value,
            self.plot_df["layer_series"],
            "r-",
            label="Current Sample",
        )

        # Plot the average if checkbox is checked
        if self._plot_average_checkbox.value:
            self.plot_widget.ax.plot(
                years,
                self.plot_df["average"],
                "g-",
                label="Average",
            )

        # Set labels and title
        self.plot_widget.ax.set_xlabel("Year")
        self.plot_widget.ax.set_ylabel("Width")
        self.plot_widget.ax.set_title("Cross-dating Comparison")
        self.plot_widget.ax.legend()
        self.plot_widget.ax.grid(True)

        # Get the min and max years for x-axis
        min_year = min(years) - 10
        max_year = max(years) + 10

        # Update x slider range but preserve values if possible
        self._x_range_slider.min = min_year
        self._x_range_slider.max = max_year

        # If the value of the slider has been initialized
        if self._x_range_slider_was_set:

            # Get current x slider values
            current_x_low, current_x_high = self._x_range_slider.value

            # Compute new x view range that preserves as much of previous view as possible
            new_x_low = (
                current_x_low
                if (current_x_low >= min_year) and (current_x_low < max_year)
                else min_year
            )
            new_x_high = (
                current_x_high
                if (current_x_high <= max_year) and (current_x_high > min_year)
                else max_year
            )

            # Only update if the previous x range isn't valid anymore
            if new_x_low != current_x_low or new_x_high != current_x_high:
                self._x_range_slider.value = (new_x_low, new_x_high)
        # Otherwise, we don't want to consider the current values as they are defaults with no meaning
        else:
            self._x_range_slider.value = (min_year, max_year)
            self._x_range_slider_was_set = True

        # Get the min and max values for y-axis
        all_values = pd.concat(
            [
                self.plot_df["reference_series"].dropna(),
                self.plot_df["layer_series"].dropna(),
            ]
        )
        min_value = all_values.min() - 10
        max_value = all_values.max() + 10

        # Update y slider range but preserve values if possible
        self._y_range_slider.min = min_value
        self._y_range_slider.max = max_value

        # If the value of the slider has been initialized
        if self._y_range_slider_was_set:

            # Get current y slider values
            current_y_low, current_y_high = self._y_range_slider.value

            # Compute new y view range that preserves as much of previous view as possible
            new_y_low = (
                current_y_low
                if (current_y_low >= min_value) and (current_y_low < max_value)
                else min_value
            )
            new_y_high = (
                current_y_high
                if (current_y_high <= max_value)
                and (current_y_high > min_value)
                else max_value
            )

            # Only update if the previous y range isn't valid anymore
            if new_y_low != current_y_low or new_y_high != current_y_high:
                self._y_range_slider.value = (new_y_low, new_y_high)
        # Otherwise, we don't want to consider the current values as they are defaults with no meaning
        else:
            self._y_range_slider.value = (min_value, max_value)
            self._y_range_slider_was_set = True

        # Set the x and y axis limits according to the sliders
        self.plot_widget.ax.set_xlim(self._x_range_slider.value)
        self.plot_widget.ax.set_ylim(self._y_range_slider.value)

        # Redraw the canvas
        self.plot_widget.figure.tight_layout()
        self.plot_widget.canvas.draw()

    def _update_plot_limits(self):
        """Update the axis limits based on the range sliders"""
        if self.plot_df is None or self.plot_df.empty:
            return

        # Get values from both sliders
        x_min, x_max = self._x_range_slider.value
        y_min, y_max = self._y_range_slider.value

        # Update both axes limits
        self.plot_widget.ax.set_xlim(x_min, x_max)
        self.plot_widget.ax.set_ylim(y_min, y_max)

        # Redraw the canvas
        self.plot_widget.canvas.draw()

    def _apply_offset_to_layer(self):
        """Apply the offset to the current input layer."""
        if self._input_layer_combo.value is None:
            return

        input_layer = self._input_layer_combo.value

        # Get the current offset value
        offset = self._offset_slider.value
        new_last_year = (
            input_layer.metadata["rings_outmost_complete_year"] + offset
        )

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

        # Reset the offset slider to 0
        self._offset_slider.value = 0
