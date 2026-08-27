import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


SCRIPT_PATH = Path(__file__).parents[2] / "turbodiffusion/inference/wan2.2_i2v_infer.py"


@pytest.fixture
def inference_module(monkeypatch):
    io_module = types.ModuleType("imaginaire.utils.io")
    io_module.save_image_or_video = lambda *args, **kwargs: None
    utils_module = types.ModuleType("imaginaire.utils")
    utils_module.log = object()
    monkeypatch.setitem(sys.modules, "imaginaire.utils", utils_module)
    monkeypatch.setitem(sys.modules, "imaginaire.utils.io", io_module)

    dataset_module = types.ModuleType("rcm.datasets.utils")
    dataset_module.VIDEO_RES_SIZE_INFO = {}
    umt5_module = types.ModuleType("rcm.utils.umt5")
    umt5_module.clear_umt5_memory = lambda: None
    umt5_module.get_umt5_embedding = lambda **kwargs: None
    vae_module = types.ModuleType("rcm.tokenizers.wan2pt1")
    vae_module.Wan2pt1VAEInterface = object
    monkeypatch.setitem(sys.modules, "rcm.datasets.utils", dataset_module)
    monkeypatch.setitem(sys.modules, "rcm.utils.umt5", umt5_module)
    monkeypatch.setitem(sys.modules, "rcm.tokenizers.wan2pt1", vae_module)

    modify_model = types.ModuleType("modify_model")
    modify_model.tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    modify_model.create_model = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "modify_model", modify_model)

    transforms = types.ModuleType("torchvision.transforms.v2")
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = types.SimpleNamespace(v2=transforms)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", torchvision.transforms)
    monkeypatch.setitem(sys.modules, "torchvision.transforms.v2", transforms)

    spec = importlib.util.spec_from_file_location("wan22_official_infer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initialize_npu_binds_selected_device_and_returns_tensor_kwargs(
    inference_module, monkeypatch
):
    assert hasattr(inference_module, "initialize_npu")

    events = []
    fake_npu = types.SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda device_id: events.append(("set_device", device_id)),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))

    result = inference_module.initialize_npu(3)

    assert events == [("set_device", 3)]
    assert result == {"device": "npu:3", "dtype": torch.bfloat16}


def test_i2v_entry_has_no_cuda_only_model_transfer():
    tree = ast.parse(SCRIPT_PATH.read_text())
    cuda_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cuda"
    ]

    assert cuda_calls == []


def test_i2v_cli_defaults_to_sla_and_rejects_sagesla(
    inference_module,
    monkeypatch,
):
    base_args = [
        "wan2.2_i2v_infer.py",
        "--high_noise_model_path",
        "high.pth",
        "--low_noise_model_path",
        "low.pth",
    ]
    monkeypatch.setattr(sys, "argv", base_args)

    args = inference_module.parse_arguments()

    assert args.attention_type == "sla"

    monkeypatch.setattr(sys, "argv", base_args + ["--attention_type", "sagesla"])
    with pytest.raises(SystemExit) as exc_info:
        inference_module.parse_arguments()

    assert exc_info.value.code == 2
