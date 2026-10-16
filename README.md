# napari-roxas-ai

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/napari-roxas-ai.svg?color=green)](https://github.com/roxas-ai/napari-roxas-ai/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-roxas-ai.svg?color=green)](https://pypi.org/project/napari-roxas-ai)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-roxas-ai.svg?color=green)](https://python.org)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://roxas-ai.github.io/napari-roxas-ai/)
[![tests](https://github.com/roxas-ai/napari-roxas-ai/workflows/tests/badge.svg)](https://github.com/roxas-ai/napari-roxas-ai/actions)
[![codecov](https://codecov.io/gh/roxas-ai/napari-roxas-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/roxas-ai/napari-roxas-ai)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-roxas-ai)](https://napari-hub.org/plugins/napari-roxas-ai)

A plugin that integrates the ROXAS AI analysis methods for quantitative wood anatomy (QWA) in the [napari](https://napari.org) platform.

> ⚠️ **Note**: This plugin is currently under active development and is not yet stable nor officially released.

📖 **Official Documentation**: [https://roxas-ai.github.io/napari-roxas-ai/](https://roxas-ai.github.io/napari-roxas-ai/)

---

## Quickstart Installation

For users familiar with Python/Conda environments:

```bash
# 1. Create and activate a dedicated environment
conda create -n roxas-ai python=3.12
conda activate roxas-ai

# 2. Install napari-roxas-ai
pip install napari-roxas-ai

# 3. Launch napari
napari
```

> 💡 **Detailed Step-by-Step Installation Guides:**  
> For complete, beginner-friendly instructions—including Miniforge/Miniconda setup for **Windows**, **macOS**, and **Linux**, as well as **GPU acceleration (CUDA / Apple Silicon MPS)** configuration—see the **[Getting Started & Installation Guide](https://roxas-ai.github.io/napari-roxas-ai/getting_started/installation/)**.

---

## Verifying Installation

To verify that the plugin is working correctly:
1. In the napari menu, go to `File > Open Sample > ROXAS AI`.
2. The first time you open a sample, sample data and model weights are downloaded automatically in the background. Progress will be displayed in the terminal.
3. Three sample layers (image, rings, cells) will open in the viewer.

---

## GPU Acceleration

ROXAS AI runs on CPU by default. For large image datasets, GPU acceleration is strongly recommended:

- **Windows & Linux (NVIDIA GPUs)**: Reinstall PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```
- **macOS (Apple Silicon M1/M2/M3/M4)**: Supported out-of-the-box via Metal Performance Shaders (MPS).
- **Settings**: In napari, go to `Plugins > ROXAS AI > ZZ - Settings` -> `processing`, enable `try_to_use_gpu` and `try_to_use_autocast`, then click `Apply`.

For full GPU prerequisites, driver requirements, and troubleshooting, consult the **[Installation Documentation](https://roxas-ai.github.io/napari-roxas-ai/getting_started/installation/#step-3-optional-gpu-acceleration)**.

---

## Contributing

Contributions are very welcome! Tests are automatically run with [tox]; please ensure that test coverage remains consistent before submitting a pull request.

### Contributor Setup

To set up a local development environment:

```bash
# 1. Create and activate environment
conda create -n roxas-ai python=3.12
conda activate roxas-ai

# 2. Install plugin in editable mode
pip install -e .

# 3. Install testing dependencies and napari plugin engine
pip install -e ".[testing]"
pip install npe2

# 4. Install pre-commit hooks for quality checks
pip install pre-commit
pre-commit install
```

### Documentation & Development Resources

- [User Documentation & Tutorials](https://roxas-ai.github.io/napari-roxas-ai/)
- [napari Plugin Developer Documentation](https://napari.org/stable/plugins/index.html)
- [napari-plugin-template](https://github.com/napari/napari-plugin-template)

---

## License

Distributed under the terms of the [GNU GPL v3.0] license, `napari-roxas-ai` is free and open source software.

## Issues

If you encounter any problems or have feature requests, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/roxas-ai/napari-roxas-ai/issues
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[tox]: https://tox.readthedocs.io/en/latest/
