# Welcome to ROXAS AI

**ROXAS AI** is an **open-source, Python-based** tool designed to streamline quantitative wood anatomy (QWA) research. By integrating state-of-the-art AI analysis methods into the [napari](https://napari.org) platform, it enables researchers to efficiently establish long, high-resolution time series of cell measurements within dated tree rings.

Currently optimized for **linear conifer anatomies**, ROXAS AI provides an end-to-end workflow—from automated image segmentation to visual cross-dating—allowing for precise dendroecological and wood anatomical studies on **Windows, macOS, and Linux**.

---

### Why ROXAS AI?

ROXAS AI was developed to modernize and expand upon the foundations of the [classical ROXAS implementation](https://roxas.wsl.ch). It directly addresses several long-standing limitations to make wood anatomy more accessible and scalable:

*   **Cross-Platform Accessibility**: Unlike the classical version, which is restricted to Windows, ROXAS AI runs natively on all major operating systems.
*   **Open Research & Powerful Viewing**: It removes the dependency on expensive third-party software (like Image-Pro Plus). Instead, by using the open-source image viewer **napari**, ROXAS AI leverages a powerful workhorse for image viewing with an intuitive and versatile editing toolbox.
*   **Modern AI Segmentation**: By leveraging specialized deep-learning neural networks (Katzenmaier et al. [2023](https://doi.org/10.1016/j.dendro.2023.126126), [2025](https://doi.org/10.3389/fpls.2025.1516635)), it overcomes the limitations of classical image segmentation, handling complex wood structures with higher accuracy and less manual intervention (see [Segmentation Models](reference/segmentation_models.md)).
*   **Ease of Use**: It reduces the requirement for expert-level configuration knowledge, making professional-grade wood anatomical analysis more intuitive.

---

## Documentation Overview
Explore the sections below to get started:

- **[Getting Started](getting_started/installation.md)**: How to install the plugin and run your first analysis.
- **[User Guide](user_guide/widgets.md)**: Detailed information about the available widgets and tools.
- **[Tutorials](tutorials/workflows.md)**: Step-by-step guides for specific tasks.
- **[Reference](reference/technical_reference.md)**: Technical specifications, output variables, and [deep-learning segmentation models](reference/segmentation_models.md).
