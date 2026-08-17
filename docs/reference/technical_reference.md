# Reference

Technical details about the files created by ROXAS AI and the numeric output variables.

## File Formats

ROXAS AI uses a standardized naming convention to organize analysis data. For a sample named `sample_01`, the following files are typically generated:

### Spatial & Image Files
- **`.scan.jpg` / `.scan.tif`** etc.: The original high-resolution scan of the wood sample; supported image formats: **JPG**, **JPEG**, **TIFF**, **TIF**, **PNG**, **BMP**, **JP2**.
- **`.cells.png`**: A label image produced by a deep-learning segmentation model where each pixel value corresponds to a unique cell ID (CID).
- **`.rings.tif`**: A label image (or vector-like representation) produced by a deep-learning based segmentation model containing the identified tree-ring boundaries.
- **`.annotated.jpg`**: (Optional) A visualization of the scan with cell and ring detections overlaid for quick review.
- **`.ReferenceSeries.jpg`**: (Optional) A visual representation of the reference chronologies used during cross-dating.

### Data & Metadata Files
- **`.metadata.json`**: A comprehensive JSON file containing sample information, imaging parameters, and all processing thresholds required to reproduce the analysis.
- **`.cells_table.csv`**: A tab-separated CSV file containing measurements for every individual cell.
- **`.rings_table.csv`**: A tab-separated CSV file containing aggregated measurements for each tree ring.

---

## Output Variables (Acronyms)

The output tables (`.cells_table.csv` and `.rings_table.csv`) use standardized acronyms. Most measurements are provided in **micrometers ($\mu\text{m}$)** or **square micrometers ($\mu\text{m}^2$)**.

### Missing Values & Error Codes
Unlike ROXAS classic, which uses numeric error codes (e.g., `-99`, `-9999`), ROXAS AI leaves these fields **blank (NA)** in the CSV output to ensure compatibility with modern data analysis tools like R and Python.

### Ring-Level Output (`.rings_table.csv`)

| Acronym | Description | Unit | Details |
| :--- | :--- | :--- | :--- |
| **RA** | Ring Area | $\text{mm}^2$ | - |
| **MRW** | Mean Ring Width | $\mu\text{m}$ | Corrected for inclined ring boundaries. |
| **CNO** | Number of Cells | count | Total cells in the tree ring. |
| **CD** | Cell Density | no./$\text{mm}^2$ | Number of cells per $\text{mm}^2$. |
| **CTA** | Cumulative Transsectional Area | $\text{mm}^2$ | Cumulative area of all counted cells. |
| **RCTA** | Conductive Area Percentage | % | CTA / RA. |
| **MLA** | Mean Lumen Area | $\mu\text{m}^2$ | Mean cell lumen area in the ring. |
| **MINLA** | Minimum Lumen Area | $\mu\text{m}^2$ | - |
| **MAXLA** | Maximum Lumen Area | $\mu\text{m}^2$ | - |
| **KH** | Hydraulic Conductance | $\text{m}^4\,\text{s}^{-1}\,\text{MPa}^{-1}$ | Theoretical (Poiseuille) adjusted for elliptical tubes. |
| **KS** | Specific Hydraulic Conductivity | $\text{m}^2\,\text{s}^{-1}\,\text{MPa}^{-1}$ | KH/RA (assuming tube length of 1m). |
| **RVGI** | Vessel Grouping Index | index | Mean cells per group (Carlquist 2001). |
| **RVSF** | Vessel Solitary Fraction | % | Fraction of solitary cells vs all cells. |
| **RGSGV** | Group Size of Grouped Cells | index | Mean size of non-solitary cell groups. |
| **CWTPI** | Cell Wall Thickness - Pith | $\mu\text{m}$ | Inner wall facing towards the pith. |
| **CWTBA** | Cell Wall Thickness - Bark | $\mu\text{m}$ | Outer wall facing towards the bark. |
| **CWTLE** | Cell Wall Thickness - Left | $\mu\text{m}$ | Measured with bark at the bottom. |
| **CWTRI** | Cell Wall Thickness - Right | $\mu\text{m}$ | Measured with bark at the bottom. |
| **CWTTAN** | Tangential Wall Thickness | $\mu\text{m}$ | Mean of pith and bark sides: ([CWTPI+CWTBA]/2). |
| **CWTRAD** | Radial Wall Thickness | $\mu\text{m}$ | Mean of left and right sides: ([CWTLE+CWTRI]/2). |
| **CWTALL** | Overall Wall Thickness | $\mu\text{m}$ | Mean of all sides: ([CWTRAD+CWTTAN]/2). |
| **RTSR** | Radial Thickness-to-Span Ratio | ratio | Mork's index; ratio between 4x CWT(tan) and radial diameter. |
| **CTSR** | Circular Thickness-to-Span Ratio | ratio | Ratio between 4x CWT(all) and area-equivalent circle diameter. |
| **DHW** | Hydraulic Diameter (W) | $\mu\text{m}$ | Mean per ring (Kolb & Sperry 1999). |
| **DHM** | Hydraulic Diameter (M) | $\mu\text{m}$ | Mean per ring (Tyree & Zimmermann 2002). |
| **DRAD** | Mean Radial Diameter | $\mu\text{m}$ | Measured bark-to-pith, corrected for inclination. |
| **DTAN** | Mean Tangential Diameter | $\mu\text{m}$ | Measured tangentially to pith, corrected for inclination. |
| **TB2** | Reinforcement Index $(t/b)^2$ | index | Hacke et al. 2001; uses the smaller of rad/tan values. |
| **CWA** | Cell Wall Area | $\mu\text{m}^2$ | Mean per ring; may include pit artefacts. |
| **RWD** | Relative Anatomical Density | ratio | Mean of CWA / (CWA+LA) per ring. |

---

### Cell-Level Output (`.cells_table.csv`)

| Acronym | Description | Unit | Details |
| :--- | :--- | :--- | :--- |
| **ID** | Sample ID | - | Unique identifier for the sample. |
| **CID** | Cell ID | - | Unique cell identifier within the sample. |
| **YEAR** | Ring Affiliation | - | Calendar year of cell formation. |
| **LA** | Cell Lumen Area | $\mu\text{m}^2$ | - |
| **XPIX** | X-Coordinate | pixels | Center of cell (0/0 is top-left corner). |
| **YPIX** | Y-Coordinate | pixels | Center of cell (0/0 is top-left corner). |
| **RADDISTR** | Radial Distance (Ring) | $\mu\text{m}$ | Distance of cell center from inner ring boundary. |
| **RRADDISTR** | Relative Radial Position | % | 0.00 (proximal boundary) to 99.99 (distal boundary). |
| **NBRNO** | Neighbor Number | count | Number of cells in the group this cell belongs to. |
| **NBRID** | Neighbor IDs | - | IDs of all cells in the same group (blank if solitary). |
| **ASP** | Aspect Ratio | ratio | Major axis / minor axis of equivalent ellipse. |
| **MAJAX** | Major Axis Deviation | degrees | Angular deviation from line towards X/Y origin. |
| **KH** | Hydraulic Conductance | $\text{m}^4\,\text{s}^{-1}\,\text{MPa}^{-1}$ | Theoretical conductance per cell. |
| **CWTPI** | Cell Wall Thickness - Pith | $\mu\text{m}$ | Facing towards the pith. |
| **CWTBA** | Cell Wall Thickness - Bark | $\mu\text{m}$ | Facing towards the bark. |
| **CWTLE** | Cell Wall Thickness - Left | $\mu\text{m}$ | Measured with bark at bottom. |
| **CWTRI** | Cell Wall Thickness - Right | $\mu\text{m}$ | Measured with bark at bottom. |
| **CWTTAN** | Tangential Wall Thickness | $\mu\text{m}$ | Mean of ([CWTPI+CWTBA]/2). |
| **CWTRAD** | Radial Wall Thickness | $\mu\text{m}$ | Mean of ([CWTLE+CWTRI]/2). |
| **CWTALL** | Overall Wall Thickness | $\mu\text{m}$ | Mean of ([CWTRAD+CWTTAN]/2). |
| **RTSR** | Radial Thickness-to-Span Ratio | ratio | Mork's index per cell. |
| **CTSR** | Circular Thickness-to-Span Ratio | ratio | Circular ratio per cell. |
| **DH** | Hydraulic Diameter | $\mu\text{m}$ | Corrected for elliptical shape (Lewis & Boose 1995). |
| **DRAD** | Radial Diameter | $\mu\text{m}$ | Bark-to-pith direction, corrected for inclination. |
| **DTAN** | Tangential Diameter | $\mu\text{m}$ | Tangential to pith, corrected for inclination. |
| **TB2** | Reinforcement Index $(t/b)^2$ | index | Cell wall reinforcement index (t/b)^2. |
| **CWA** | Cell Wall Area | $\mu\text{m}^2$ | Wall area per cell; may include pit artefacts. |
| **RWD** | Relative Anatomical Density | ratio | Cell-level CWA / (CWA+LA). |

---

## Theoretical Hydraulic Conductance ($K_h$)
The theoretical hydraulic conductance is approximated by Poiseuille's law and adjusted for elliptical tubes (Nonweiler 1975):

$$ KH = \frac{LA \cdot m^2}{\nu \cdot k} $$

Where:
- **LA**: Lumen area ($\text{m}^2$)
- **$\nu$**: Viscosity of water ($1.002 \cdot 10^{-3}\,\text{Pa}\cdot\text{s}$ at 20°C)
- **m**: Mean hydraulic radius ($\pi ab / C$)
- **k**: Geometry coefficient ($4 / (1 + \sqrt{1 - e^4})$)
- **e**: Eccentricity of the ellipse

---

## The Metadata File (`.metadata.json`)

The metadata file ensures traceability and reproducibility of your analysis. It stores four main categories of information:

### 1. Sample & Project Information
- **`sample_name`**: The unique identifier for the wood sample.
- **`sample_type`**: The type of wood (e.g., `conifer`).
- **`meas_geometry`**: The measurement layout (e.g., `linear`).
- **`rings_outmost_complete_year`**: The calendar year assigned to the most recent complete tree ring.

### 2. Imaging & Spatial Metadata
- **`spatial_resolution`**: The conversion factor from pixels to micrometers ($\mu\text{m}/\text{pixel}$). This is critical for all subsequent physical measurements.
- **`scan_size`**: The dimensions of the original image in pixels (Height, Width).
- **`scan_info`**: Detailed technical info about the image, including the detected **DPI** (Dots Per Inch).
- **`scan_exif`**: Metadata extracted directly from the image file (e.g., software used for scanning, capture date).

### 3. Processing & Traceability
- **`sw_version`**: The exact version of ROXAS AI used for the analysis.
- **`cells_segmentation_model`** & **`rings_segmentation_model`**: The specific AI model filenames (`.pth`) used for detection.
- **`meas_created_at`**: A timestamp of when the final measurement table was generated.

### 4. Analysis Parameters & Quality Control
These parameters define how cell walls are measured and how clusters are handled:

- **`cluster_dbl_cwt_threshold`**: Threshold for detecting cell wall clusters.
- **`relwidth_cwt_integration`**: The relative width along the cell wall used for integrating thickness measurements.
- **`opposite_cwt_ratio_limit`** & **`adjacent_cwt_ratio_limit`**: Quality control factors used to flag or correct suspicious wall thickness measurements.

---

## Global Settings (`settings.json`)

While the `.metadata.json` is unique to each sample, the `settings.json` file stores your global preferences and plugin configurations. 

**Important**: One of the core goals of ROXAS AI is to be accessible. Unlike classical systems, **no expert knowledge is required** to configure these settings. For most users, the default values are optimized for professional results and do not need to be changed.

The settings are organized into several logical blocks:

### 1. File Extensions
Defines how ROXAS AI recognizes and names files (e.g., using `.scan.jpg` for images and `.cells_table.csv` for measurements). This ensures consistency across your projects.

### 2. Processing & GPU
Allows you to toggle hardware acceleration. By setting `try_to_use_gpu` to `true`, you can significantly speed up the AI segmentation process if a compatible graphics card is available.

### 3. Visualization (Vectorization & Rasterization)
Controls how data appears in the napari viewer:
- **Colors**: Default colors for cells (e.g., lime), rings (e.g., red), and the color sequence used to distinguish adjacent tree rings.
- **Line Widths**: Adjusts how thick the boundaries appear on your screen for better visibility.

### 4. Measurement Defaults
Contains the default thresholds for anatomical analysis. These values are used to populate the metadata for new samples:

*   **`cluster_dbl_cwt_threshold`**: Threshold for detecting cell wall clusters (physiological grouping via pits).
*   **`cells_smoothing_kernel_size`**: Size of the kernel used for smoothing cell boundaries.
*   **`relwidth_cwt_integration`**: The relative width along the cell wall used for thickness calculations.
*   **`cells_tangential_angle`**: Default angle for tangential orientation.
*   **`lower_limit_cwt_iqr_multiplier`** & **`upper_limit_cwt_iqr_multiplier`**: Multipliers for identifying outliers in cell wall thickness using the Interquartile Range (IQR).
*   **`opposite_cwt_ratio_limit`**: Quality control factor comparing wall thickness on opposite sides of a cell.
*   **`adjacent_cwt_ratio_limit`**: Quality control factor comparing wall thickness on adjacent sides of a cell.

### 5. Metadata Field Definitions
Defines the structure of the input forms you see in the plugin widgets, ensuring that required information like `sample_type` or `spatial_resolution` is always captured correctly.
