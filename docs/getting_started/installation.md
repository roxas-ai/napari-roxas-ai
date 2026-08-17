# Getting Started

Welcome to the `napari-roxas-ai` plugin! This guide will help you set up your environment and install the plugin to start your wood anatomy analysis.

## Installation

### 1. Environment Setup

It is strongly recommended to create a dedicated Python environment for `napari-roxas-ai` to avoid dependency conflicts.

1.  **Install a package manager**: If you don't have it already, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
2.  **Create a new environment**:
    ```bash
    conda create -n roxas-ai python=3.12
    conda activate roxas-ai
    ```

### 2. Install the Plugin

Install `napari-roxas-ai` via [pip](https://pypi.org/project/pip/):

```bash
pip install napari-roxas-ai
```

### 3. Launching the Plugin

Once installed, you can launch the napari viewer with the plugin enabled:

```bash
napari
```

---

## Verifying Installation

To check if the plugin is working correctly:
1.  Launch **napari**.
2.  Go to `File > Open Sample > ROXAS AI`.
3.  **Note**: The first time you open a sample, it may take some time as sample data and model weights are being downloaded. Progress will be logged in the terminal.
4.  After the downloads, you should see three layers (image, rings, and cells) open in the viewer.

---

## GPU Support

While the plugin runs on the CPU by default, enabling GPU acceleration is highly recommended for faster inference on large images.

### NVIDIA GPUs (Windows & Linux)

The default installation uses a CPU-only build of PyTorch. To enable NVIDIA GPU support, reinstall PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
*Note: This command supports RTX 20-series and newer. You only need an up-to-date NVIDIA driver.*

### macOS (Apple Silicon)

On M1/M2/M3 chips, PyTorch uses **MPS acceleration**. No additional drivers are needed. Note that currently, MPS acceleration is primarily leveraged for the cell segmentation model.

### Enabling GPU in Settings

After installing the correct version of PyTorch:
1.  In napari, go to `Plugins > ROXAS AI > ZZ - Settings`.
2.  Under the `processing` section, set `try_to_use_gpu` and `try_to_use_autocast` to `true`.
3.  Restart napari for the changes to take effect.

---

## Your First Analysis

Once installation is verified, you are ready to start. You can either use the sample data provided or load your own images via the **Loading** widget.
