# Deep-Learning Segmentation Models

Technical reference on the deep-learning neural network architectures, boundary regression formulation, tiled inference pipeline, and model checkpoint management in **ROXAS AI**.

---

## Overview

Traditional quantitative wood anatomy (QWA) workflows often rely on manual thresholding and morphological filtering, which can be sensitive to staining variations, compression wood, cutting artifacts, or resin deposits. **ROXAS AI** replaces these manual thresholds with specialized deep-learning models trained specifically on conifer wood micro-sections and surface scans:

1. **Cell Lumina Semantic Segmentation (`linear_cells_model`)**: Delineates tracheid cell lumina across whole-slide cross-sections.
2. **Tree-Ring Boundary Regression (`linear_rings_model.pth` / `LinearRingModel_v3_iterative.pt`)**: Detects annual growth boundaries using an iterative regression formulation along the radial axis.

All models run locally on your system using PyTorch and PyTorch Lightning. Inference seamlessly leverages hardware acceleration (NVIDIA CUDA or Apple Silicon Metal Performance Shaders / MPS) when available, falling back to multi-core CPU execution.

---

## Cell Segmentation Model

The cell segmentation model automates the detection and lumen delineation of individual tracheids in high-resolution conifer cross-sections.

```
+-----------------------------------------------------------------------------+
|                           CELL INFERENCE PIPELINE                           |
|                                                                             |
|  [ Full Scan Image ]                                                        |
|         │                                                                   |
|         ▼                                                                   |
|  [ Reflection Padding (112 px) ]                                            |
|         │                                                                   |
|         ▼                                                                   |
|  [ Tiled Patches: 1024x1024 px, Stride: 800 px ]                            |
|         │                                                                   |
|         ▼                                                                   |
|  [ ResNet-50 Encoder + U-Net Decoder (smp.Unet) ]                           |
|         │                                                                   |
|         ▼                                                                   |
|  [ Argmax / Binary Class Probability Map ]                                  |
|         │                                                                   |
|         ▼                                                                   |
|  [ Center Patch Stitching (800x800 px unpadded region) ]                    |
|         │                                                                   |
|         ▼                                                                   |
|  [ Connected-Component Labeling + Border Component Removal ]                |
|         │                                                                   |
|         ▼                                                                   |
|  [ Final Binary Mask / Cell Labels Layer (.cells) ]                         |
+-----------------------------------------------------------------------------+
```

### Architecture & Parameters

- **Architecture**: **U-Net** semantic segmentation network built with a **ResNet-50** encoder backbone (initialized via `segmentation_models_pytorch`).
- **Input Channels & Normalization**: Standard 3-channel RGB image patches normalized via ResNet-50 pre-processing statistics.
- **Output Classes**: Binary classification (Background vs. Cell Lumen).
- **Inference Mode**: Evaluated in `eval()` mode with automatic mixed precision (`torch.autocast`) when GPU acceleration is active.

### Tiled Patch Inference & Reflection Padding

High-resolution anatomical scans frequently exceed tens of thousands of pixels in height and width. To handle massive images with bounded GPU memory usage while preventing edge artifacts:

1. **Patch Dimensions**: Patches of dimension $1024 \times 1024\text{ px}$ are extracted with a sliding stride of $800\text{ px}$.
2. **Reflection Padding**: The full image is padded in `reflect` mode with a margin of:
   $$\text{pad\_size} = \frac{\text{patch\_size} - \text{stride}}{2} = \frac{1024 - 800}{2} = 112\text{ px}$$
3. **Margin Discarding & Seamless Stitching**: Predictions generated on the $1024 \times 1024\text{ px}$ patch are sliced to retain only the central $800 \times 800\text{ px}$ region, completely discarding boundary distortion zones before stitching into the final canvas.
4. **Border Component Post-Processing**: Cells touching the outer edges of the entire image canvas are filtered out (`remove_border_touching_components`) to prevent partial, truncated cell measurements.
5. **Instance Labeling**: Binary predictions are labeled into discrete cell instances using connected components (`skimage.measure.label`).

---

## Tree-Ring Boundary Segmentation Model

Annual ring boundaries define the temporal framework for dendrochronological dating and ring-width series extraction.

```
+-----------------------------------------------------------------------------+
|                           RING INFERENCE PIPELINE                           |
|                                                                             |
|  [ Full Scan Image / Radial Microsection ]                                  |
|         │                                                                   |
|         ▼                                                                   |
|  [ Iterative Boundary Regression Network (PackageImporter) ]                 |
|         │                                                                   |
|         ▼                                                                   |
|  [ Sequential Boundary Polylines (Bark -> Pith) ]                           |
|         │                                                                   |
|         ▼                                                                   |
|  [ Coordinate Reordering (Left-to-Right) ]                                  |
|         │                                                                   |
|         ▼                                                                   |
|  [ Horizontal Completion to Canvas Margins (x=0 to x=Width) ]                |
|         │                                                                   |
|         ▼                                                                   |
|  [ Top Incomplete Boundary Prepended (Dummy Row 0) ]                         |
|         │                                                                   |
|         ▼                                                                   |
|  [ Annual Ring Polygon Bands & Raster Labels (.rings) ]                     |
+-----------------------------------------------------------------------------+
```

### Iterative Boundary Regression Formulation

Unlike conventional semantic segmentation models that classify pixels independently (often leading to broken, fragmented, or bleeding ring lines), ROXAS AI adopts an **iterative regression formulation**:

- **Continuous Coordinate Trajectories**: The model regresses ordered coordinate trajectories $P_k = \{(y_1, x_1), (y_2, x_2), \dots, (y_n, x_n)\}$ representing each annual boundary line sequentially from the bark toward the pith.
- **Topological Continuity**: Because boundaries are represented as parametric curves, lines remain unbroken across faint latewood zones, narrow rings, or local staining blemishes.
- **Linear Conifer Adaptation**: While the generalized framework supports multi-species circular geometries (Katzenmaier et al. 2025), the integrated model checkpoint is optimized for linear radial microsections common in microtome-based quantitative wood anatomy.

### Post-Processing & Geometry Construction

Once boundary polyline trajectories are predicted:

1. **Left-to-Right Sorting**: Boundary point sequences are sorted from the left margin to the right margin (`rearrange_coordinates`).
2. **Horizontal Completion**: Boundary endpoints are extended horizontally to the image boundaries ($x = 0$ and $x = W - 1$) using `horizontal_rings_completion`, ensuring complete division into annual polygons without self-intersections.
3. **Dummy Top Boundary**: A horizontal reference line is prepended at $y = 0$ (row index `0`) to mark the outer, incomplete growth ring (assigned `enabled = False`).
4. **Rasterization & Ring Colormap**: The enclosed polygonal areas between consecutive boundaries are rasterized into integer labels, rendered using a high-contrast categorical ring colormap.

### Targeted Re-Segmentation Below a Given Year

In the **Edit Rings** widget (`6 - Edit rings`), users can re-run the deep-learning model starting from any selected calendar year. The model re-segments all boundaries below that anchor year while preserving manually validated boundaries above it, eliminating the need to re-process the entire image when correcting early-stage errors.

---

## Model Weights & Automatic Management

ROXAS AI automatically handles model checkpoint discovery, remote downloads, integrity checks, and local caching.

| Model Identifier | File Name | Format | Primary Task |
| :--- | :--- | :--- | :--- |
| **Cell Segmentation** | `linear_cells_model.pth` | PyTorch State Dict / TorchScript | Tracheid lumen semantic segmentation |
| **Ring Segmentation** | `linear_rings_model.pth` | PyTorch Package (`torch.package`) | Initial annual ring boundary regression |
| **Ring Re-run Model** | `LinearRingModel_v3_iterative.pt` | PyTorch Package (`torch.package`) | Iterative boundary regression below edited ring |

### Background Download & Local Checkpoints

- **Automatic Asset Detection**: Upon opening segmentation widgets or running batch inference, ROXAS AI checks local directories (`_models/_cells` and `_models/_rings`).
- **GitHub Release Storage**: If checkpoints are missing or incomplete, the asset manager (`check_assets_and_download`) fetches `cells_models.zip` and `rings_models.zip` directly from official GitHub releases.
- **Atomic Extraction & Tracking**: Archives are downloaded into temporary directories and extracted atomically. A `.asset_id` marker tracks the release version to prevent duplicate downloads.
- **Offline Reliability**: Once downloaded, model checkpoints reside locally and do not require active internet access.

### Hardware Acceleration & Resource Management

- **Device Selection**: Controlled via the `processing.try_to_use_gpu` setting in `ZZ - Settings`.
  - **NVIDIA GPUs**: Leverages CUDA acceleration with tensor cores and mixed precision.
  - **Apple Silicon**: Uses Metal Performance Shaders (`mps`) backend on M-series chips.
  - **CPU Fallback**: Seamless fallback on systems without dedicated GPU hardware.
- **Thread Safety**: Multithreaded execution sets `torch.set_num_threads(1)` within background worker threads to prevent OpenMP runtime conflicts across Qt event loops.

---

## Scientific Literature & References

If you use ROXAS AI in academic publications, please cite the following foundational papers describing the deep-learning models:

- **Katzenmaier et al. (2023) — Conifer Cell Segmentation**:  
  Katzenmaier, M., Sainte Fare Garnot, V., Björklund, J., Schneider, L., Wegner, J. D., & von Arx, G. (2023). Towards ROXAS AI: deep learning for faster and more accurate conifer cell analysis. *Dendrochronologia*, 81, 126126.  
  [https://doi.org/10.1016/j.dendro.2023.126126](https://doi.org/10.1016/j.dendro.2023.126126)  
  *Introduces the deep-learning semantic segmentation framework for automated conifer tracheid lumen extraction and anatomical validation.*

- **Katzenmaier et al. (2025) — Tree-Ring Boundary Regression**:  
  Katzenmaier, M., Garnot, V. S. F., Wegner, J. D., & von Arx, G. (2025). Towards ROXAS AI: automatic multi-species ring boundaries segmentation as regression in anatomical images. *Frontiers in Plant Science*, 16, 1516635.  
  [https://doi.org/10.3389/fpls.2025.1516635](https://doi.org/10.3389/fpls.2025.1516635)  
  *Presents the iterative boundary regression formulation for tree-ring boundary detection in wood anatomical images.*
