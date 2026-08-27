import contextlib
import importlib

import torch


class FakeVAEModel:
    def eval(self):
        return self

    def requires_grad_(self, requires_grad):
        return self

    def to(self, **kwargs):
        return self


def patch_vae_construction(monkeypatch, wan_vae):
    tensor_devices = []
    model_devices = []
    real_tensor = torch.tensor

    def tensor_on_cpu(*args, **kwargs):
        tensor_devices.append(kwargs.pop("device", None))
        return real_tensor(*args, **kwargs)

    def fake_video_vae(*args, **kwargs):
        model_devices.append(kwargs["device"])
        return FakeVAEModel()

    monkeypatch.setattr(wan_vae.torch, "tensor", tensor_on_cpu)
    monkeypatch.setattr(wan_vae, "_video_vae", fake_video_vae)
    return tensor_devices, model_devices


def test_wan_vae_uses_npu_by_default(monkeypatch):
    wan_vae = importlib.import_module("rcm.tokenizers.wan2pt1")
    tensor_devices, model_devices = patch_vae_construction(monkeypatch, wan_vae)

    vae = wan_vae.WanVAE(is_amp=False)

    assert vae.device == "npu"
    assert tensor_devices == ["npu", "npu"]
    assert model_devices == ["npu"]


def test_wan_vae_amp_uses_npu_autocast(monkeypatch):
    wan_vae = importlib.import_module("rcm.tokenizers.wan2pt1")
    patch_vae_construction(monkeypatch, wan_vae)
    autocast_devices = []

    def fake_autocast(device_type, dtype):
        autocast_devices.append(device_type)
        return contextlib.nullcontext()

    monkeypatch.setattr(wan_vae.torch.amp, "autocast", fake_autocast)

    wan_vae.WanVAE(is_amp=True)

    assert autocast_devices == ["npu"]
