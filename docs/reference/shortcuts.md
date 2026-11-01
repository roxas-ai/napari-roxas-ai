# Keyboard & Mouse Shortcuts Reference

Complete reference of keyboard shortcuts and mouse bindings available in **ROXAS AI** for high-throughput image navigation, manual vector/raster curation, and workflow dock management in napari.

!!! tip "Shortcut Availability & Activation"
    - **Launching via `napari-roxas`**: Running the custom launcher `napari-roxas` installs all navigation and workflow shortcuts immediately upon startup and opens the **Manual Editing** dock container.
    - **Launching via standard `napari`**: When starting via plain `napari`, shortcuts and duplicate dock protection are installed and activated as soon as any ROXAS AI widget or menu action is opened.

---

## Complete Shortcut Summary Table

| Category | Action | Key(s) / Mouse Binding | Scope / Active Context | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Navigation** | Pan Up | <kbd>W</kbd> or <kbd>↑</kbd> | Global Viewer | Pans the viewport upward by 200 pixels. |
| **Navigation** | Pan Down | <kbd>S</kbd> or <kbd>↓</kbd> | Global Viewer | Pans the viewport downward by 200 pixels. |
| **Navigation** | Pan Left | <kbd>A</kbd> or <kbd>←</kbd> | Global Viewer | Pans the viewport leftward by 200 pixels. |
| **Navigation** | Pan Right | <kbd>D</kbd> or <kbd>→</kbd> | Global Viewer | Pans the viewport rightward by 200 pixels. |
| **Navigation** | Fast Pan | <kbd>Shift</kbd> + (<kbd>W</kbd>/<kbd>A</kbd>/<kbd>S</kbd>/<kbd>D</kbd> or <kbd>↑</kbd>/<kbd>←</kbd>/<kbd>↓</kbd>/<kbd>→</kbd>) | Global Viewer | Doubles panning speed (400 pixels per step). |
| **Navigation** | Zoom Out | <kbd>Q</kbd> | Global Viewer | Decreases magnification by a step of $0.25$. |
| **Navigation** | Zoom In | <kbd>E</kbd> | Global Viewer | Increases magnification by a step of $0.25$. |
| **Widget Groups** | Manual Editing Group | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>M</kbd> *(macOS: <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>M</kbd>)* | Global Viewer | Closes existing docks and opens the unified Manual Editing sidebar. |
| **Widget Groups** | Batch Operations Group | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd> *(macOS: <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd>)* | Global Viewer | Closes existing docks and opens the unified Batch Operations sidebar. |
| **Widget Groups** | Close All Docks | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>W</kbd> *(macOS: <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>W</kbd>)* | Global Viewer | Closes all open dock widgets in the napari window. |
| **Cell Editing** | Adjust Brush Size | <kbd>Ctrl</kbd> + Mouse Wheel *(macOS: <kbd>Cmd</kbd> + Wheel)* | `5 - Edit cells` (Raster Mode) | Dynamically resizes the painting/erasing brush (1 to 100 px). |
| **Cell Editing** | Switch Brush / Eraser | <kbd>Ctrl</kbd> + Double-Click *(macOS: <kbd>Cmd</kbd> + Double-Click)* | `5 - Edit cells` (Raster Mode) | Toggles between the paint brush and eraser tool. |
| **Cell Editing** | Delete Lasso Cells | <kbd>Delete</kbd> or <kbd>Backspace</kbd> | `5 - Edit cells` (Vector Mode) | Deletes all cell polygons intersecting the drawn lasso area. |
| **Cell Editing** | Undo Last Edit | <kbd>Ctrl</kbd> + <kbd>Z</kbd> *(macOS: <kbd>Cmd</kbd> + <kbd>Z</kbd>)* | `5 - Edit cells` (Active Editing) | Reverts the last stroke or vector modification (up to 50 undo steps). |
| **Ring Editing** | Fine Nudge Boundary (1 px) | <kbd>Ctrl</kbd> + <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>←</kbd> / <kbd>→</kbd> *(macOS: <kbd>Cmd</kbd> + Arrows)* | `6 - Edit rings` (Selected Boundaries) | Nudges all selected ring boundary vertices by 1 pixel in the chosen direction. |
| **Ring Editing** | Fast Nudge Boundary (10 px) | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>←</kbd> / <kbd>→</kbd> | `6 - Edit rings` (Selected Boundaries) | Nudges all selected ring boundary vertices by 10 pixels in the chosen direction. |
| **Ring Editing** | Delete Lasso Vertices | <kbd>Delete</kbd> or <kbd>Backspace</kbd> | `6 - Edit rings` (Lasso Mode) | Deletes vertices from topmost boundary enclosed by the lasso area. |
| **Ring Editing** | Undo Last Edit | <kbd>Ctrl</kbd> + <kbd>Z</kbd> *(macOS: <kbd>Cmd</kbd> + <kbd>Z</kbd>)* | `6 - Edit rings` (Active Editing) | Reverts the last boundary edit, deletion, or affine move (up to 50 undo steps). |

---

## Detailed Mechanics by Category

### 1. Viewport Navigation & Magnification

Wood anatomical cross-sections can contain millions of pixels. Single-key navigation shortcuts allow rapid, continuous inspection without switching between active drawing/selection tools:

- **Single-Key Panning (<kbd>W</kbd>/<kbd>A</kbd>/<kbd>S</kbd>/<kbd>D</kbd> or Arrow Keys)**: Shifts the visible camera viewport by **200 screen pixels** in the corresponding direction.
- **Fast Panning (<kbd>Shift</kbd> + Navigation Keys)**: Doubles the movement step to **400 screen pixels**, useful for traversing long radial stem increments.
- **Incremental Zoom (<kbd>Q</kbd> / <kbd>E</kbd>)**: Adjusts camera magnification by $\pm 0.25$ zoom increments while keeping the viewport centered.

---

### 2. Workflow Dock Groups & Sidebar Management

ROXAS AI organizes its 11 tools into unified, scrollable dock containers to keep the workspace clean:

- **Manual Editing Group (<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>M</kbd> / macOS: <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>M</kbd>)**:
  Opens the primary interactive analysis workflow containing `2/4 - Load image(s)`, `3A - Detect cells & rings`, `5 - Edit cells`, `6 - Edit rings`, and `9A - Measure cells & rings`.
- **Batch Operations Group (<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd> / macOS: <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>B</kbd>)**:
  Opens the high-throughput batch pipeline containing `1 - Prepare project images`, `3B - Batch detect cells & rings`, and `9B - Batch measure & update`.
- **Close All Docks (<kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>W</kbd> / macOS: <kbd>Cmd</kbd> + <kbd>Shift</kbd> + <kbd>W</kbd>)**:
  Clears all open dock widgets to maximize screen real estate for visual inspection.

---

### 3. Cell Editing Shortcuts (`5 - Edit cells`)

In the **Edit Cells** widget, shortcuts streamline both pixel-level raster touch-ups and vector polygon curation:

- **Dynamic Brush Resizing (<kbd>Ctrl</kbd> + Mouse Wheel / macOS: <kbd>Cmd</kbd> + Wheel)**:
  - While using the Paint Brush or Eraser in Raster mode, holding <kbd>Ctrl</kbd> and scrolling dynamically adjusts brush diameter between 1 and 100 pixels.
  - Step scaling is proportional: $\text{step} = \max(1.0, 0.15 \times \text{brush\_size})$, providing single-pixel precision at small diameters and rapid scaling at large sizes.
  - Standard napari dimension slider scrolling is temporarily suppressed during editing to prevent accidental slice jumps.
- **Brush / Eraser Quick Toggle (<kbd>Ctrl</kbd> + Double-Click / macOS: <kbd>Cmd</kbd> + Double-Click)**:
  - Holding <kbd>Ctrl</kbd> and double-clicking on the canvas instantly swaps between painting and erasing modes without needing to click toolbar icons.
- **Lasso Polygon Deletion (<kbd>Delete</kbd> or <kbd>Backspace</kbd>)**:
  - In Vector mode with **Lasso Select** active, drawing one or more closed loops and pressing <kbd>Delete</kbd> removes all cell polygons that intersect the lasso selection.
- **Undo Stack (<kbd>Ctrl</kbd> + <kbd>Z</kbd> / macOS: <kbd>Cmd</kbd> + <kbd>Z</kbd>)**:
  - Maintains a history of up to 50 operations. In vector mode, it restores exact shape geometries and properties; in raster mode, it triggers napari's label brush undo history.

---

### 4. Tree-Ring Editing Shortcuts (`6 - Edit rings`)

In the **Edit Rings** widget, shortcuts facilitate precise alignment of vector polylines:

- **Fine Boundary Nudging (1 px)**:
  - Select one or more boundary polylines using the Direct Select tool and press <kbd>Ctrl</kbd> + <kbd>↑</kbd> / <kbd>↓</kbd> / <kbd>←</kbd> / <kbd>→</kbd> (or <kbd>Cmd</kbd> + Arrows on macOS).
  - Shifts all vertices of selected boundaries by exactly 1 pixel, ideal for sub-lumen alignment along latewood cell walls.
- **Fast Boundary Nudging (10 px)**:
  - Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Arrow keys to shift selected boundary vertices by 10 pixels per step.
- **Topmost Lasso Vertex Deletion (<kbd>Delete</kbd> or <kbd>Backspace</kbd>)**:
  - When **Lasso Select** is active in ring editing mode, drawing a selection loop and pressing <kbd>Delete</kbd> removes vertices located inside the lasso.
  - **Topmost Layer Priority Rule**: When multiple boundary polylines overlap closely within the lasso area, vertices are removed exclusively from the topmost boundary (highest shape index), preventing accidental deletion of underlying ring lines.
- **Undo History (<kbd>Ctrl</kbd> + <kbd>Z</kbd> / macOS: <kbd>Cmd</kbd> + <kbd>Z</kbd>)**:
  - Restores prior boundary polylines, calendar year metadata assignments, deleted vertex markers, and affine transform modifications across a 50-step history stack.

---

## Cross-Platform Modifier Key Reference

| Action / Shortcut | Windows / Linux | macOS Equivalent |
| :--- | :--- | :--- |
| **Workflow Groups** | <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + Key | <kbd>Cmd</kbd> + <kbd>Shift</kbd> + Key |
| **Brush Sizing** | <kbd>Ctrl</kbd> + Mouse Wheel | <kbd>Cmd</kbd> + Mouse Wheel / Trackpad Scroll |
| **Brush / Eraser Toggle** | <kbd>Ctrl</kbd> + Double-Click | <kbd>Cmd</kbd> + Double-Click |
| **Boundary Nudging** | <kbd>Ctrl</kbd> (+ <kbd>Shift</kbd>) + Arrows | <kbd>Cmd</kbd> (+ <kbd>Shift</kbd>) + Arrows |
| **Deletion** | <kbd>Delete</kbd> or <kbd>Backspace</kbd> | <kbd>Delete</kbd> or <kbd>Backspace</kbd> |
| **Undo** | <kbd>Ctrl</kbd> + <kbd>Z</kbd> | <kbd>Cmd</kbd> + <kbd>Z</kbd> |
