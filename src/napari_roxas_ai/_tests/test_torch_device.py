"""
Tests for choosing the torch device the segmentation models run on.

The backends are faked so that the order CUDA -> XPU -> MPS -> CPU can be
checked on any machine, including ones without an NVIDIA or Intel GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from napari_roxas_ai._utils._torch_device import select_torch_device


def _fake_backends(monkeypatch, available, missing=()):
    for name in ("cuda", "xpu", "mps"):
        if name in missing:
            monkeypatch.delattr(torch, name, raising=False)
        else:
            backend = SimpleNamespace(is_available=lambda n=name: n in available)
            monkeypatch.setattr(torch, name, backend)


@pytest.mark.parametrize(
    "available, expected",
    [
        ({"cuda", "xpu", "mps"}, "cuda"),
        ({"xpu", "mps"}, "xpu"),
        ({"mps"}, "mps"),
        (set(), "cpu"),
    ],
)
def test_first_available_backend_is_chosen(monkeypatch, available, expected):
    _fake_backends(monkeypatch, available)
    assert select_torch_device(True) == expected


def test_cpu_when_gpu_is_not_wanted(monkeypatch):
    _fake_backends(monkeypatch, {"cuda", "xpu", "mps"})
    assert select_torch_device(False) == "cpu"


def test_missing_backend_module_is_skipped(monkeypatch):
    # torch builds before 2.4 have no torch.xpu
    _fake_backends(monkeypatch, {"mps"}, missing=("xpu",))
    assert select_torch_device(True) == "mps"
