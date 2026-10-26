# Keyboard & Mouse Shortcuts Reference

**ROXAS AI** includes an extensive set of keyboard and mouse shortcuts designed to accelerate wood anatomical inspection, manual curation of cell and ring geometries, and workflow management within napari.

!!! tip "Shortcut Availability & Activation"
    - **Launching with `napari-roxas`**: Starting the application via the dedicated launcher command `napari-roxas` installs all shortcuts immediately upon startup and opens the **Manual Editing** widget group docked in the sidebar.
    - **Launching with plain `napari`**: When starting via standard `napari`, ROXAS AI shortcuts are automatically installed and activated as soon as any ROXAS AI menu command or widget is opened for the first time in that session.

---

## Complete Shortcut Summary

| Category | Action | Key(s) / Mouse Binding | Scope / Availability | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Navigation** | Pan Up | <kbd>W</kbd> or <kbd>↑</kbd> | Always in Viewer | Pans the viewport upward by 200 pixels. |
| **Navigation** | Pan Down | <kbd>S</kbd> or <kbd>↓</kbd> | Always in Viewer | Pans the viewport downward by 200 pixels. |
| **Navigation** | Pan Left | <kbd>A</kbd> or <kbd>←</kbd> | Always in Viewer | Pans the viewport leftward by 200 pixels. |
| **Navigation** | Pan Right | <kbd>D</kbd> or <kbd>→</kbd> | Always in Viewer | Pans the viewport rightward by 200 pixels. |
| **Navigation** | Fast Pan | <kbd>Shift</kbd> + (<kbd>W</kbd>/<kbd>A</kbd>/<kbd>S</kbd>/<kbd>D</kbd> or <kbd>↑</kbd>/<kbd>←</kbd>/<kbd>↓</kbd>/<kbd>→</kbd>) | Always in Viewer | Doubles panning speed (400 pixels per step). |
| **Navigation** | Zoom Out | <kbd>Q</kbd> | Always in Viewer | Decreases magnification by a step of $0.25$. |
| **Navigation** | Zoom In | <kbd>E</kbd> | Always in Viewer | Increases magnification by a step of $0.25$. |
| **Widget Groups** | Open Manual Editing Group | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>M</kbd> | Always in Viewer | Closes current docks and opens the unified Manual Editing group sidebar. |
| **Widget Groups** | Open Batch Operations Group | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd> | Always in Viewer | Closes current docks and opens the unified Batch Operations group sidebar. |
| **Widget Groups** | Close All Dock Widgets | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>W</kbd> | Always in Viewer | Closes all open sidebar dock widgets in the napari window. |
| **Cell Editing** | Adjust Brush Size | <kbd>Ctrl</kbd> + Mouse Wheel *(macOS: <kbd>Cmd</kbd> + Wheel)* | `5 - Edit cells` (Raster Mode) | Dynamically resizes the painting/erasing brush (1 to 100 px). |
| **Cell Editing** | Switch Brush / Eraser | <kbd>Ctrl</kbd> + Double-Click *(macOS: <kbd>Cmd</kbd> + Double-Click)* | `5 - Edit cells` (Raster Mode) | Instantly toggles between the paint brush and eraser tool. |
| **Cell Editing** | Delete Lasso Selection | <kbd>Delete</kbd> or <kbd>Backspace</kbd> | `5 - Edit cells` (Vector Mode) | Deletes all cell polygons touching the drawn lasso area. |
| **Cell Editing** | Undo Last Change | <kbd>Ctrl</kbd> + <kbd>Z</kbd> | `5 - Edit cells` (Active Editing) | Reverts the last stroke or vector modification (up to 50 undo steps). |
| **Ring Editing** | Fine Nudge Boundary (1 px) | <kbd>Ctrl</kbd> + <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>←</kbd> / <kbd>→</kbd> | `6 - Edit rings` (Selected Boundaries) | Nudges all selected ring boundary vertices by 1 pixel in the chosen direction. |
| **Ring Editing** | Fast Nudge Boundary (10 px) | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>←</kbd> / <kbd>→</kbd> | `6 - Edit rings` (Selected Boundaries) | Nudges all selected ring boundary vertices by 10 pixels in the chosen direction. |
| **Ring Editing** | Delete Lasso Vertices | <kbd>Delete</kbd> or <kbd>Backspace</kbd> | `6 - Edit rings` (Lasso Mode) | Deletes vertices from topmost boundary enclosed by the lasso area. |
| **Ring Editing** | Undo Last Change | <kbd>Ctrl</kbd> + <kbd>Z</kbd> | `6 - Edit rings` (Active Editing) | Reverts the last boundary edit, deletion, or affine move (up to 50 undo steps). |

---

## Detailed Shortcut Descriptions

### 1. Viewport Navigation & Zoom (Global)
These single-key navigation shortcuts allow continuous scanning across large wood microsections without losing active drawing or editing tools:
- **WASD / Arrow Keys**: Smoothly pan the viewport in increments of 200 screen pixels.
- **<kbd>Shift</kbd> + WASD / Arrow Keys**: Rapidly pan across the sample in increments of 400 screen pixels.
- **<kbd>Q</kbd> / <kbd>E</kbd>**: Incrementally zoom out (<kbd>Q</kbd>) or zoom in (<kbd>E</kbd>) by $0.25$ zoom increments.

### 2. Workflow Widget Groups & Dock Management
- **<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>M</kbd>**: Opens the **Manual Editing** workflow container (`2/4 - Load image(s)`, `3A - Detect cells & rings`, `5 - Edit cells`, `6 - Edit rings`, `9A - Measure cells & rings`).
- **<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd>**: Opens the **Batch Operations** workflow container (`1 - Prepare project images`, `3B - Batch detect cells & rings`, `9B - Batch measure & update`).
- **<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>W</kbd>**: Clears the viewer sidebar by closing all open dock widgets.

### 3. Cell Layer Editor (`5 - Edit cells`)
- **Brush Size Adjustment (<kbd>Ctrl</kbd> + Mouse Wheel / Trackpad Scroll)**:
  While painting or erasing in Raster mode, holding <kbd>Ctrl</kbd> (or <kbd>Cmd</kbd> on macOS) and scrolling the mouse wheel adjusts the brush diameter between 1 and 100 pixels. The step size scales proportionally with the current brush size for fine control at small diameters and rapid scaling at large sizes. Standard napari dimension slider scrolling is temporarily disabled to prevent accidental slice shifting.
- **Brush / Eraser Toggle (<kbd>Ctrl</kbd> + Double-Click)**:
  Holding <kbd>Ctrl</kbd> and double-clicking on the image instantly swaps between the paint brush and the eraser tool, facilitating quick alternating between filling missing lumens and erasing false detections.
- **Lasso Deletion (<kbd>Delete</kbd> or <kbd>Backspace</kbd>)**:
  In Vector mode with **Lasso Select** enabled, pressing <kbd>Delete</kbd> or <kbd>Backspace</kbd> deletes all cell polygons that intersect the drawn lasso loop(s).
- **Undo (<kbd>Ctrl</kbd> + <kbd>Z</kbd>)**:
  In Vector mode, ROXAS AI captures geometry snapshots in an undo stack (up to 50 steps), allowing full reversal of additions, vertex edits, and lasso deletions. In Raster mode, it triggers napari's label brush undo history.

### 4. Ring Layer Editor (`6 - Edit rings`)
- **Fine and Fast Nudging**:
  When one or more ring boundaries are selected in the vector shapes layer:
  - <kbd>Ctrl</kbd> + <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>←</kbd> / <kbd>→</kbd> shifts selected boundaries by **1 pixel**.
  - <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>←</kbd> / <kbd>→</kbd> shifts selected boundaries by **10 pixels**.
- **Lasso Vertex Deletion (<kbd>Delete</kbd> or <kbd>Backspace</kbd>)**:
  When **Lasso Select** is checked, pressing <kbd>Delete</kbd> or <kbd>Backspace</kbd> deletes only the vertices from the topmost ring boundary enclosed by the lasso polygon, leaving underlying or overlapping boundaries untouched.
- **Undo (<kbd>Ctrl</kbd> + <kbd>Z</kbd>)**:
  Restores the prior boundary coordinates, calendar year metadata, deleted vertex markers, and affine transform states (up to 50 history steps).
