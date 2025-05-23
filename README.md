# napari-roxas-ai

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/napari-roxas-ai.svg?color=green)](https://github.com/roxas-ai/napari-roxas-ai/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-roxas-ai.svg?color=green)](https://pypi.org/project/napari-roxas-ai)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-roxas-ai.svg?color=green)](https://python.org)
[![tests](https://github.com/roxas-ai/napari-roxas-ai/workflows/tests/badge.svg)](https://github.com/roxas-ai/napari-roxas-ai/actions)
[![codecov](https://codecov.io/gh/roxas-ai/napari-roxas-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/roxas-ai/napari-roxas-ai)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-roxas-ai)](https://napari-hub.org/plugins/napari-roxas-ai)

A plugin that integrates the ROXAS AI analysis methods for quantitative wood anatomy in the napari platform

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

### Environment setup
It's recommended to create a dedicated Python environment for napari-roxas-ai:

1. Install Miniconda if you don't have it already: [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

2. Create a new environment:
```bash
conda create -n roxas-ai python=3.12
conda activate roxas-ai
```

### Installation
Install `napari-roxas-ai` via [pip]:

```bash
pip install napari-roxas-ai
```

### Launching the plugin
Once installed, you can launch napari with the roxas-ai plugin:

```bash
napari
```

### Verifying installation
To check if the plugin is working correctly:
1. Go to `File > Open Sample > ROXAS AI` in the napari interface.
2. The first time you open a sample, it may take some time as sample data and model weights are being downloaded. Progress will be logged in the terminal.
3. After the downloads, a sample made of three layers should open in the viewer

### GPU Support
If you want to use GPU acceleration for model inference:

1. Ensure you have the proper GPU drivers and CUDA installed for your system:
   - [NVIDIA CUDA Installation Guide Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
   - [NVIDIA CUDA Installation Guide Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

2. Enable GPU support in the napari-roxas-ai settings within the napari interface.

3. You may need to reinstall PyTorch with CUDA support for your specific hardware:
   Visit the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to find the appropriate installation command for your setup.

## Contributing

Contributions are very welcome. Tests are automatically run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

### Contributor Installation

In order to contribute to the development of the plugin the installation can be done as follows:
1. Create an environment
```bash
conda create -n roxas-ai python=3.12
conda activate roxas-ai
```
2. In the cloned / forked plugin directory, install the plugin dependencies
```bash
pip install -e .
```

3. Install the testing dependencies, as well as the napari plugin engine
```bash
pip install -e ".[testing]"
pip install npe2
```

4. Install pre-commit for quality checks
```bash
pip install pre-commit
pre-commit install
```

### Documentation: Plugin Template and Development
You can find more information on the plugin template on the [napari-plugin-template repository](https://github.com/napari/napari-plugin-template).
You can find more information on plugin contributions and how to create plugins on the [plugins section of the napari documentation](https://napari.org/dev/plugins/index.html).

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"napari-roxas-ai" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/roxas-ai/napari-roxas-ai/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
