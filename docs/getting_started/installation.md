# Installation & Setup Guide

Welcome to **ROXAS AI**! This guide walks you through setting up a clean Python environment, installing the `napari-roxas-ai` plugin, and configuring optional GPU acceleration for fast deep-learning inference.

---

## Step 1: Install a Package Manager & Open Terminal

We strongly recommend installing **Miniforge** (recommended for scientific computing with `conda-forge` defaults) or **Miniconda** to manage your Python environment.

=== "Windows"

    1. **Download the installer:**
        - **Miniforge (Recommended):** Download the [Miniforge Windows 64-bit Installer](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe).
        - **Miniconda (Alternative):** Download the [Miniconda Windows 64-bit Installer](https://docs.anaconda.com/miniconda/).
    2. **Run the installer:** Follow the standard setup wizard.
    3. **Open the prompt:**
        - Open the Windows Start menu and search for **Miniforge Prompt** (or **Anaconda Prompt** if using Miniconda).
        - Open it to run the subsequent commands.

=== "macOS"

    1. **Download the installer:**
        - **Apple Silicon (M1/M2/M3/M4):** Download [Miniforge3-MacOSX-arm64.pkg](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.pkg) or the [Miniconda Apple Silicon Installer](https://docs.anaconda.com/miniconda/).
        - **Intel Macs:** Download [Miniforge3-MacOSX-x86_64.pkg](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.pkg) or the [Miniconda Intel Installer](https://docs.anaconda.com/miniconda/).
    2. **Run the installer package** and follow the prompts.
    3. *Alternative (Command-line installation via Terminal):*
        ```bash
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"
        bash Miniforge3-MacOSX-$(uname -m).sh
        ```
    4. **Open Terminal:** Open `Applications > Utilities > Terminal`.

=== "Linux"

    1. **Download the installer in Terminal:**
        - **x86_64 (64-bit Intel/AMD):**
          ```bash
          wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
          bash Miniforge3-Linux-x86_64.sh
          ```
        - **ARM64 / aarch64:**
          ```bash
          wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
          bash Miniforge3-Linux-aarch64.sh
          ```
    2. Follow the installer prompts and allow it to initialize Conda for your shell.
    3. Open a new Terminal window afterwards.

---

## Step 2: Create & Activate the Environment

Create a clean Python 3.12 environment named `roxas-ai` and activate it:

```bash
conda create -n roxas-ai python=3.12
conda activate roxas-ai
```

!!! tip "Always activate your environment"
    Whenever you open a new terminal session to run ROXAS AI, make sure to activate the environment first using `conda activate roxas-ai`.

---

## Step 3: (Optional) GPU Acceleration

While ROXAS AI functions fully on CPU, enabling GPU acceleration significantly speeds up deep-learning segmentation on large high-resolution images.

=== "Windows & Linux (NVIDIA GPUs)"

    By default, PyPI installs a CPU-only build of PyTorch on Windows and Linux. To leverage your NVIDIA GPU:

    1. **Install CUDA-enabled PyTorch:**
        ```bash
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu128
        ```
        *This single command supports all recent NVIDIA GPUs (RTX 20-series, 30-series, 40-series, and 50-series / Blackwell). The CUDA runtime is bundled with the wheels, so no separate CUDA toolkit installation is needed—only an up-to-date NVIDIA display driver.*

        *(Note: If your system uses an older display driver that cannot be updated, you can install the CUDA 12.6 build instead by replacing `cu128` with `cu126`).*

    2. **Verify GPU / CUDA availability:**
        ```python
        python -c "
        import torch
        print('CUDA available:', torch.cuda.is_available())
        print('CUDA version:', torch.version.cuda)
        print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
        "
        ```
        *For a functional GPU installation, `CUDA available:` should output `True` and list your GPU device name.*

    3. **Troubleshooting:**
        If `CUDA available:` reports `False` after installation (e.g., if a CPU-only PyTorch build was previously cached or installed), cleanly remove PyTorch and reinstall:
        ```bash
        pip uninstall torch torchvision torchaudio -y
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu128
        ```

    *(For supported AMD GPUs on Linux, consult the [PyTorch ROCm installation guide](https://pytorch.org/get-started/locally/)).*

=== "Windows & Linux (Intel GPUs)"

    !!! warning "Experimental"
        Intel GPU (XPU) support in PyTorch is still a prototype. It covers Intel Arc A- and B-Series graphics cards and Intel Core Ultra processors with integrated Arc graphics on Windows 11 and Ubuntu. See the [PyTorch Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) for the full list.

    1. **Install the Intel GPU driver:**
        Follow Intel's driver installation guide linked in the PyTorch Intel GPU guide above. No separate oneAPI installation is needed.

    2. **Install XPU-enabled PyTorch:**
        ```bash
        pip install "torch>=2.5,<2.9" "torchvision<0.24" \
          --index-url https://download.pytorch.org/whl/xpu
        ```
        *The version limits keep PyTorch inside the range ROXAS AI supports, so installing ROXAS AI afterwards does not replace it with a CPU-only build.*

    3. **Verify XPU availability:**
        ```python
        python -c "
        import torch
        print('XPU available:', torch.xpu.is_available())
        print('GPU:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'None')
        "
        ```
        *For a functional installation, `XPU available:` should output `True` and list your GPU device name. If it reports `False`, check the driver installation.*

=== "macOS (Apple Silicon & Intel)"

    - **Apple Silicon (M1/M2/M3/M4):**
      PyTorch leverages Apple's **Metal Performance Shaders (MPS)** backend natively. No separate CUDA or driver installation is necessary. The standard ROXAS AI package includes the appropriate PyTorch build.
      
      To verify MPS availability on Apple Silicon:
      ```python
      python -c "
      import torch
      print('MPS available:', torch.backends.mps.is_available())
      print('MPS built:', torch.backends.mps.is_built())
      "
      ```
      *When `MPS available: True` is displayed, ROXAS AI will accelerate cell segmentation using your Apple GPU.*

    - **Intel-based Macs:**
      GPU acceleration is not supported on Intel Macs; ROXAS AI will run automatically using the CPU.

---

## Step 4: Install ROXAS AI

Install the latest released version of `napari-roxas-ai` via `pip`:

```bash
pip install napari-roxas-ai
```

---

## Step 5: Launch & Verify

1. **Launch napari with ROXAS AI:**
   ```bash
   napari-roxas
   ```
   This starts napari with the ROXAS AI shortcuts installed and the **Manual Editing** widget group docked. Plain `napari` works as well, the shortcuts then become available once a ROXAS AI widget is opened.
2. **Open sample data:**
   - In the napari menu, navigate to `File > Open Sample > ROXAS AI`.
   - Three layers (**Image**, **Rings**, and **Cells**) will load into the viewer.

!!! info "Automatic Background Model Downloads"
    Pre-trained neural network weights (`.pth` / `.pt` files) and sample data are automatically downloaded in the background when first accessed and cached locally for all future sessions. Download progress is logged in your terminal window.

3. **Enable GPU acceleration in Settings (if applicable):**
   - In napari, go to `Plugins > ROXAS AI > ZZ - Settings`.
   - Expand the **`processing`** section.
   - Check the boxes for **`try_to_use_gpu`** and **`try_to_use_autocast`**.
   - Click **`Apply`**.
   - *The settings take effect immediately (no restart required), though any segmentation or measurement run already in progress will complete with the settings it was started with.*

---

## Summary Checklist

- [x] **Miniforge / Miniconda** installed and opened in the terminal.
- [x] Dedicated Python 3.12 environment created and activated (`conda activate roxas-ai`).
- [x] *(Optional)* GPU PyTorch installed and tested (`CUDA available: True` / `MPS available: True`).
- [x] `napari-roxas-ai` installed via `pip`.
- [x] `napari-roxas` launched and sample data loaded successfully.
- [x] *(Optional)* `try_to_use_gpu` enabled in `ZZ - Settings`.

!!! warning "Important note"
    Do not install packages into the environment while napari or another Python process is actively running in it. Always perform package installations and updates from a dedicated terminal prompt before launching napari.
