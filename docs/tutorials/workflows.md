# Tutorials & Workflows

Step-by-step guides for quantitative wood anatomy (QWA) and dendrochronological analysis using ROXAS AI in napari.

---

## 1. Quickstart: 5-Minute Single Sample Walkthrough

This fast-track walkthrough gets you from raw image to measured anatomical traits using the bundled conifer sample.

```mermaid
flowchart LR
    A["Load Conifer Sample"] --> B["Detect Cells & Rings (3A)"]
    B --> C["Review Annotations (5 & 6)"]
    C --> D["Measure Traits (9A)"]
```

### Step 1: Open the Built-in Sample Data
1. Launch **napari**.
2. Navigate to the top menu: **File > Open Sample > ROXAS AI > Conifer Sample Data**.
3. ROXAS AI automatically loads the conifer cross-section scan along with pre-configured spatial metadata ($\mu\text{m}/\text{px}$).

![Screenshot: Built-in sample loaded in napari](../assets/screenshots/tutorials/quickstart_01_loaded_sample.png)

*Figure 1: Conifer sample dataset loaded into the napari viewer.*

### Step 2: Run Deep Learning Detection
1. Open the plugin menu: **Plugins > ROXAS AI > 3A - Detect cells & rings (individual image)**.
2. Ensure the active image is selected in the **Sample Image** dropdown.
3. Select the segmentation models:
   - **Cells Model**: `linear_cells_model`
   - **Rings Model**: `linear_rings_model`
4. Click **Run Segmentation**. Deep-learning inference generates two new layers: `.cells` (cell lumen labels) and `.rings` (ring boundary labels).

![Screenshot: Automated DL detection result overlay](../assets/screenshots/tutorials/quickstart_02_dl_segmentation.png)

*Figure 2: Automated cell lumen and tree-ring boundary segmentation overlay.*

### Step 3: Quick Visual Verification
1. Toggle layer visibility in the left napari layer list to inspect the overlay against the anatomical scan.
2. Use **Plugins > ROXAS AI > 6 - Edit rings** or **5 - Edit cells** if minor adjustments are needed.

### Step 4: Extract Cell & Ring Measurements
1. Open **Plugins > ROXAS AI > 9A - Measure cells & rings (individual image)**.
2. Select the active sample and click **Run Measurements**.
3. Measurement output tables (`.cells_table.txt` and `.rings_table.txt`) are created in the project folder with comprehensive anatomical metrics (lumen area, cell wall thickness, Mork's index, ring widths).

![Screenshot: Extracted anatomical measurement tables](../assets/screenshots/tutorials/quickstart_03_measurements_done.png)

*Figure 3: Measurement completion with generated tabular files.*

---

## 2. Standard Workflow: Dendroclimatic Tree-Ring & Cell Anatomy Time Series

This comprehensive, step-by-step workflow describes the standard protocol for generating publication-quality dendroclimatic and wood anatomical time series from raw microsections or high-resolution surface scans.

```mermaid
flowchart TD
    P1["Phase 1: Project Setup & Image Preparation"] --> P2["Phase 2: Automated Detection (3A)"]
    P2 --> P3["Phase 3: Tree-Ring Boundary Curation (6)"]
    P3 --> P4["Phase 4: Visual Cross-Dating (7)"]
    P4 --> P5["Phase 5: Cell Lumen Curation (5)"]
    P5 --> P6["Phase 6: Save Annotations (8)"]
    P6 --> P7["Phase 7: Trait Extraction (9A)"]
```

---

### Phase 1: Project Setup & Image Preparation

1. Open **Plugins > ROXAS AI > 0 - Define project directory** and choose the root directory where your analysis files will reside.
2. Open **Plugins > ROXAS AI > 1 - Prepare project images for analysis**.
3. The preparation assistant scans your folder for raw images (`.tif`, `.jpg`, `.png`, etc.) and opens the **Metadata Dialog**:
   - **Resolution ($\mu\text{m}/\text{px}$)**: Enter the calibrated micrometer-per-pixel value of your optical microscope or scanner.
   - **Anatomy / Growth Type**: Select `conifer`.
   - **Geometry**: Select `linear` for standard radial increment cores or microsections.
   - **Outermost Ring Year**: Set the calendar year of the outermost complete ring (e.g., sampling year `2023`).
4. *(Optional)* Click **Start Processing Crossdating Files** to import reference ring-width chronologies (`.rwl`, `.tuc`, or tabular `.txt`/`.csv`).
5. Click **Prepare Images**. The images are standardized to `<sample>.scan.<ext>` and initialized with `<sample>.metadata.json`.

![Screenshot: Image preparation metadata setup](../assets/screenshots/tutorials/std_01_image_preparation.png)

*Figure 4: Setting spatial resolution and calendar dating parameters in the image preparation dialog.*

---

### Phase 2: Loading & Automated Segmentation

1. Open **Plugins > ROXAS AI > 2/4 - Load image(s)** and load your prepared sample.
2. Open **Plugins > ROXAS AI > 3A - Detect cells & rings (individual image)**.
3. Choose the appropriate neural network models and click **Run Segmentation**.
4. Both cell lumen masks and ring boundary contours will populate as interactive napari layers.

![Screenshot: Loaded segmentation layers](../assets/screenshots/tutorials/std_02_segmentation_layers.png)

*Figure 5: Automated cell and ring predictions visualized in napari.*

---

### Phase 3: Tree-Ring Editing & Boundary Curation

Accurate ring boundaries are essential because all radial cell positions and annual time series metrics depend on them. Open **Plugins > ROXAS AI > 6 - Edit rings**.

![Screenshot: Tree-ring boundary editing widget](../assets/screenshots/tutorials/std_03_ring_editing_overview.png)

*Figure 6: Ring layer editor interface with polyline modification tools.*

#### Best Practices for Ring Boundary Tracing

- Ring boundaries must extend completely across the tangential width of the analyzed sample track.
- Boundaries should align precisely along the latewood-earlywood transition (the outer wall of the terminal latewood tracheid row).
- Ring lines must not self-intersect or cross neighboring ring boundaries.

#### Recommended Vertex Editing Strategies

Editing ring polylines in napari requires an efficient vertex manipulation strategy:

!!! warning "Vertex Deletion and Addition Sequence"
    When deleting existing vertices and immediately adding new vertices within the same uninterrupted editing action, polygon/polyline indexing can connect the new vertices in an unexpected sequence (such as creating self-intersecting loops or back-tracking lines).
    
    **Rule**: Always confirm/apply your changes after deleting vertices before starting a new edit action to insert additional vertices.

##### The "Two-Point Reset" Strategy (Fastest Reconstruction)
When an automatically detected ring boundary has significant geometric deviations or artifacts, attempting to move individual vertices or perform mixed delete/insert operations often takes excessive time. Instead, use the **Two-Point Reset**:

1. **Delete the defective ring boundary** or clear its internal vertices.
2. **Draw a simple straight line** connecting the start and end points at the lateral sample margins.
3. **Switch to vertex addition/insertion mode**: Starting from one margin and progressing systematically toward the other, insert intermediate vertices along the actual anatomical boundary.
4. **Fine-tune**: Only place as many vertices as necessary to capture curvature and local boundary transitions.

This approach ensures minimal vertex count, eliminates loop artifacts, and is substantially faster than cleaning up messy automated traces.

![Screenshot: Two-point reset strategy for ring boundary reconstruction](../assets/screenshots/tutorials/std_03b_ring_twopoint_strategy.png)

*Figure 7: The Two-Point Reset strategy: (A) draw initial straight polyline; (B) sequentially insert critical intermediate vertices along the terminal latewood boundary.*

#### Handling Missing, Wedging, and False Rings

- **Wedging Rings**: When a ring tapers off laterally, trace the boundary up to the point of convergence and terminate cleanly according to standard dendrochronological rules.
- **Intra-Annual Density Fluctuations (False Rings)**: If the neural network identified a false latewood band, delete the boundary line. The calendar years of subsequent rings will automatically update in sequence.
- **Locally Absent / Missing Rings**: Insert a new polyline at the expected anatomical position. ROXAS AI will re-index all ring labels and calendar years accordingly.

---

### Phase 4: Visual Cross-Dating & Ring Synchronization

To verify ring counts and calendar year assignments against existing master chronologies or sister cores, open **Plugins > ROXAS AI > 7 - Visual cross-dating**.

![Screenshot: Visual cross-dating interface](../assets/screenshots/tutorials/std_04_crossdating_plotter.png)

*Figure 8: Visual cross-dating plotter showing sample ring widths (blue) synchronized against reference series (orange).*

1. **Select Reference Series**: Choose the active sample or master chronology from the series list.
2. **Inspect Ring-Width Plots**: Examine the curve overlay in raw or log-transformed scale. Characteristic pointer years (exceptionally narrow or wide rings) provide anchor points.
3. **Adjust Dating**:
   - If a missing ring is detected, return to **6 - Edit rings**, insert the missing boundary, and refresh the plot.
   - If an offset exists, update the outermost year or shift the series interactively.
4. **Confirm Calibration**: Once the growth patterns match reference marker years, the chronological alignment is validated.

---

### Phase 5: High-Accuracy Lumen & Cell Wall Editing

With verified ring boundaries in place, refine cell lumen annotations. Open **Plugins > ROXAS AI > 5 - Edit cells**.

![Screenshot: Cell layer editor tools](../assets/screenshots/tutorials/std_05_cell_editor_tools.png)

*Figure 9: Cell layer editor displaying raster brush/eraser controls and vector polygon conversion.*

#### Raster vs. Vector Editing Modes

ROXAS AI offers two complementary editing modes for cell lumina:

| Mode | Best Used For | Tooling & Mechanics |
| :--- | :--- | :--- |
| **Raster Mode** | Rapid mask cleaning, separating merged cells, filling voids | Pixel brush, eraser, paint bucket. Direct modification of the integer label mask. |
| **Vector Mode** | High-precision boundary adjustments, split lines, polygon inspection | Polygon contour representation with vertex manipulation. |

#### Editing Techniques

1. **Separating Merged Lumens**:
   - In Raster mode, switch to the **Eraser** tool with a small radius (1–2 px) and draw a dividing cut through the shared wall.
   - Run the connected component refresh to split into distinct labels.
2. **Removing Debris & Resin Canals**:
   - Use the **Lasso Selection** tool in Vector mode to encircle non-tracheid structures or resin ducts, then press `Delete` to remove all enclosed objects simultaneously.
3. **Restoring Overlooked Tracheids**:
   - Use the **Brush** tool in Raster mode or draw a new polygon in Vector mode over faint or damaged cell walls.

![Screenshot: Lasso selection and multi-cell cleanup](../assets/screenshots/tutorials/std_05b_lasso_cleanup.png)

*Figure 10: Using lasso selection to batch-remove resin canal cells and preparation artifacts.*

---

### Phase 6: Saving Verified Annotations

1. Open **Plugins > ROXAS AI > 8 - Save cells & rings editing**.
2. Select the active sample and confirm layer selections.
3. Click **Save Layers**.
4. The system updates the synchronized files on disk:
   - `<sample>.cells.png`: 16-bit or 32-bit cell label mask.
   - `<sample>.rings.tif`: Vectorized ring boundary masks and calibrated ring identifiers.
   - `<sample>.metadata.json`: Updated metadata including user edits and verification timestamps.

![Screenshot: Save editing dialog](../assets/screenshots/tutorials/std_06_save_layers.png)

*Figure 11: Saving finalized annotation layers to disk.*

---

### Phase 7: Anatomical Traits Extraction & Chronology Export

1. Open **Plugins > ROXAS AI > 9A - Measure cells & rings (individual image)**.
2. Configure measurement parameters:
   - **Cell Wall Thickness (CWT) Method**: Continuous Wavelet Transform (CWT) filtering parameters.
   - **Mork's Index Threshold**: Default $4.0$ for distinguishing earlywood from latewood tracheids.
   - **Radial Sector Bands**: Number of tangential sub-sectors for radial profile averaging.
3. Click **Run Measurements**.

#### Generated Output Tables

The measurement engine outputs two primary text tables in the sample directory:

1. **`<sample>.cells_table.txt`**: Cell-level records:
   - `ID`, `X_CAL`, `Y_CAL`: Spatial coordinates ($\mu\text{m}$).
   - `AREA`: Lumen area ($\mu\text{m}^2$).
   - `CWT_RAD`, `CWT_TAN`, `CWT_ALL`: Radial, tangential, and mean cell wall thickness ($\mu\text{m}$).
   - `MORK`: Mork's index value and earlywood/latewood classification.
   - `RING_YEAR`: Calendar year of the enclosing tree ring.
   - `REL_RAD_POS`: Normalized radial position within the ring ($0.0 = \text{earlywood start}, 1.0 = \text{ring boundary}$).
2. **`<sample>.rings_table.txt`**: Ring-level summary records:
   - `YEAR`: Calendar year.
   - `RING_WIDTH`: Measured ring width ($\mu\text{m}$).
   - `NUM_CELLS`: Total count of measured tracheids.
   - `MEAN_LUMEN_AREA`: Mean lumen area per annual ring ($\mu\text{m}^2$).
   - `MEAN_CWT`: Mean annual cell wall thickness ($\mu\text{m}$).
   - `EARLYWOOD_WIDTH`, `LATEWOOD_WIDTH`: Zonal width measurements.

![Screenshot: Final measurement tables and visual summary](../assets/screenshots/tutorials/std_07_measurement_summary.png)

*Figure 12: Measurement summary and exported tabular time series.*

---

## 3. Production Workflow: Batch Processing Multi-Sample Studies

When analyzing dozens or hundreds of wood sections, ROXAS AI allows automated batch processing to minimize hands-on computer time.

```mermaid
flowchart TD
    B1["Batch Preparation (Widget 1)"] --> B2["Headless Deep Learning (Widget 3B)"]
    B2 --> B3["Targeted Visual Curation (Widgets 2, 5, 6, 7, 8)"]
    B3 --> B4["Batch Trait Extraction (Widget 9B)"]
```

### Phase 1: Batch Preparation & Standardization
1. Organize raw scans into your designated project folder.
2. Launch **Plugins > ROXAS AI > 1 - Prepare project images for analysis**.
3. Choose the directory and batch-assign spatial resolution and metadata across all matching images.

![Screenshot: Batch preparation interface](../assets/screenshots/tutorials/batch_01_prep_dialog.png)

*Figure 13: Batch image preparation and metadata assignment.*

### Phase 2: Headless Deep Learning Segmentation
1. Open **Plugins > ROXAS AI > 3B - Batch detect cells & rings**.
2. Select your project directory and desired segmentation models.
3. Click **Run Batch Segmentation**.
4. The background queue sequentially processes each image, generating `.cells.png` and `.rings.tif` masks without requiring manual interaction for each image.

![Screenshot: Batch segmentation progress window](../assets/screenshots/tutorials/batch_02_progress.png)

*Figure 14: Batch segmentation execution and progress tracking.*

### Phase 3: Targeted Quality Control
1. Open **Plugins > ROXAS AI > 2/4 - Load image(s)**.
2. Iterate through processed samples.
3. Perform targeted curation only on samples with irregular anatomy, reaction wood, or damaged sections using Widgets **5**, **6**, **7**, and **8**.

### Phase 4: Automated Batch Measurement Extraction
1. Open **Plugins > ROXAS AI > 9B - Batch measure cells & rings**.
2. Select the project directory and desired anatomical trait calculation options.
3. Click **Run Batch Measurements**.
4. Measurement tables (`.cells_table.txt` and `.rings_table.txt`) are generated for all samples across the entire project directory.

![Screenshot: Batch measurement completion](../assets/screenshots/tutorials/batch_03_done.png)

*Figure 15: Batch measurement extraction over multiple sample directories.*

---

## 4. Troubleshooting & Quality Control Cookbook

### Problem 1: Low-Contrast or Unevenly Stained Cross-Sections
- **Symptom**: Automated deep-learning segmentation under-segments cell lumina in dark or poorly illuminated regions.
- **Solution**:
  1. In the napari layer controls for the active image layer, adjust **contrast limits** and **gamma** to enhance cell wall visibility.
  2. In **5 - Edit cells**, use the raster brush with auto-thresholding enabled to quickly paint over faint lumina.

### Problem 2: Wedging Rings and Ambiguous Boundaries
- **Symptom**: Ring boundaries converge or become indistinct due to micro-wedging or localized compression.
- **Solution**:
  1. Inspect neighboring radial tracks to trace the anatomical trajectory of the ring boundary.
  2. Use the **Two-Point Reset Strategy** (see [Phase 3](#the-two-point-reset-strategy-fastest-reconstruction)) to place a clean reference guide line.
  3. Validate against the reference chronology in **7 - Visual cross-dating**.

### Problem 3: Ring Orientation and Lateral Border Clipping
- **Symptom**: Ring boundaries fail to register cell assignments near the outer image edges.
- **Solution**:
  - Ensure that ring boundary polylines completely cross the lateral edges of the analyzed cell area. Polylines that stop short will exclude bordering tracheids from annual ring assignment.

### Problem 4: Resolving Vertex Loops in Ring Tracing
- **Symptom**: A ring boundary creates an unexpected loop or cross-pattern after adding vertices.
- **Solution**:
  - Delete the misordered vertices, click outside or toggle the edit tool to commit the clean state, and then re-insert the vertices sequentially from one end of the polyline to the other.
