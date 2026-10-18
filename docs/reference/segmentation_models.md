# Deep-Learning Segmentation Models

Technical details about the deep-learning neural network architectures, inference pipelines, and scientific literature behind automated segmentation in ROXAS AI.

---

## Overview

ROXAS AI replaces classical threshold- and edge-based image processing with deep learning models trained specifically on wood anatomical micro-sections and surface scans. This enables robust, automated detection of both individual tracheid cell lumina and annual tree-ring boundaries across challenging image conditions (e.g., uneven staining, compression wood, or anatomical distortions).

Models run locally on your machine using PyTorch and PyTorch Lightning, with automatic hardware acceleration when a compatible GPU (CUDA or Apple Silicon MPS) is detected.

---

## Cell Segmentation Model

The cell segmentation model automatically detects and delineates individual tracheid cell lumina in high-resolution conifer cross-sections.

### Architecture & Pipeline
- **Network Architecture**: A **U-Net** semantic segmentation architecture equipped with a **ResNet-50** encoder backbone (via `segmentation_models_pytorch`).
- **Tiled Patch Inference**: High-resolution scans (which often span tens of thousands of pixels) are processed in overlapping tiles ($1024 \times 1024\text{ px}$ with an $800\text{ px}$ stride).
- **Reflection Padding**: To eliminate boundary and edge distortions at patch margins, inputs are padded in `reflect` mode ($112\text{ px}$ margin), and only the unpadded central regions are stitched back into the final sample mask.
- **Instance Labeling**: The binary lumen predictions are transformed into unique Cell IDs (CID) via connected-component labeling (`skimage.measure.label`).

### Scientific Reference
The cell segmentation methodology and validation are described in:

> **Katzenmaier M, Sainte Fare Garnot V, Björklund J, Schneider L, Wegner JD, von Arx G. 2023.** Towards ROXAS AI: deep learning for faster and more accurate conifer cell analysis. *Dendrochronologia* 81:126126. [https://doi.org/10.1016/j.dendro.2023.126126](https://doi.org/10.1016/j.dendro.2023.126126)

---

## Tree-Ring Boundary Segmentation Model

The tree-ring boundary model identifies annual ring boundaries in wood samples, defining annual growth increments for dendrochronological analysis.

### Methodology & Formulation
- **Iterative Boundary Regression**: Rather than performing standard pixel-wise segmentation, the model detects ring boundary trajectories sequentially along the radial growth axis, formulating boundary line identification as an iterative regression problem.
- **Post-Processing & Ordering**: Identified boundary trajectories are sorted continuously from bark to pith, extended across the sample width, and converted into labeled annual ring bands.
- **Relation to Published Model**: The ring boundary model implemented in ROXAS AI is based on the deep-learning regression framework introduced in Katzenmaier et al. (2025). Note that while the published paper demonstrates the approach for angiosperms and circular sample geometries, the model integrated in this version of the app is adapted for linear conifer anatomies.

### Scientific Reference
The underlying ring boundary regression methodology is described in:

> **Katzenmaier M, Garnot VSF, Wegner JD, von Arx G. 2025.** Towards ROXAS AI: automatic multi-species ring boundaries segmentation as regression in anatomical images. *Frontiers in Plant Science* 16:1516635. [https://doi.org/10.3389/fpls.2025.1516635](https://doi.org/10.3389/fpls.2025.1516635)

---

## Model Weights & Automatic Management

- **Default Checkpoints**: Pre-trained weights for the cell model (`linear_cells_model.pth`) and ring model (`LinearRingModel_v3_iterative.pt` / `linear_rings_model.pth`) are managed automatically.
- **Automatic Downloads**: If model weights are not found in the local installation directory, ROXAS AI automatically downloads them in the background from remote asset storage upon first use.
- **Caching**: Once downloaded, model checkpoints are cached locally and reused for all subsequent single-image and batch-processing runs.

---

## References

- **Katzenmaier et al. 2023**: Katzenmaier, M., Sainte Fare Garnot, V., Björklund, J., Schneider, L., Wegner, J. D., & von Arx, G. (2023). Towards ROXAS AI: deep learning for faster and more accurate conifer cell analysis. *Dendrochronologia*, 81:126126. [https://doi.org/10.1016/j.dendro.2023.126126](https://doi.org/10.1016/j.dendro.2023.126126)
- **Katzenmaier et al. 2025**: Katzenmaier, M., Garnot, V. S. F., Wegner, J. D., & von Arx, G. (2025). Towards ROXAS AI: automatic multi-species ring boundaries segmentation as regression in anatomical images. *Frontiers in Plant Science*, 16:1516635. [https://doi.org/10.3389/fpls.2025.1516635](https://doi.org/10.3389/fpls.2025.1516635)
