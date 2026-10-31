"""
Selection of the torch device the segmentation models run on.
"""

# Checked in this order: NVIDIA (CUDA), Intel (XPU), Apple Silicon (MPS)
GPU_BACKENDS = ("cuda", "xpu", "mps")


def select_torch_device(try_to_use_gpu: bool) -> str:
    """
    Return the first available GPU backend, or "cpu" if none is available
    or try_to_use_gpu is False.
    """
    # Imported here so that importing this module does not load torch
    import torch

    if not try_to_use_gpu:
        return "cpu"
    for backend_name in GPU_BACKENDS:
        # Older torch builds may lack the backend module (e.g. torch.xpu < 2.4)
        backend = getattr(torch, backend_name, None)
        if backend is not None and backend.is_available():
            return backend_name
    return "cpu"
