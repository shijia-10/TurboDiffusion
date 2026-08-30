import importlib
import sys
import types

import torch


def test_umt5_module_import_does_not_initialize_cuda(monkeypatch):
    """Importing the text encoder must not touch CUDA before NPU setup."""
    module_name = "rcm.utils.umt5"
    sys.modules.pop(module_name, None)

    def fail_if_cuda_is_queried():
        raise AssertionError("UMT5 import queried the CUDA runtime")

    monkeypatch.setattr(torch.cuda, "current_device", fail_if_cuda_is_queried)

    module = importlib.import_module(module_name)

    assert module.UMT5EncoderModel is not None


def test_umt5_checkpoint_is_loaded_directly_from_cpu(monkeypatch):
    umt5 = importlib.import_module("rcm.utils.umt5")
    checkpoint = {"encoder.weight": torch.tensor([1.0])}

    def load_checkpoint(path, map_location):
        if map_location != "cpu":
            raise AssertionError(f"checkpoint loaded on {map_location}")
        return checkpoint

    class FakeModel:
        def __init__(self):
            self.loaded_state = None

        def load_state_dict(self, state, assign):
            self.loaded_state = (state, assign)

    monkeypatch.setattr(umt5.distributed, "is_rank0", lambda: True)
    monkeypatch.setattr(umt5.distributed, "sync_model_states", lambda model, src: None)
    monkeypatch.setattr(torch, "load", load_checkpoint)
    model = FakeModel()

    result = umt5.load_model_torch(model, "umt5.pth")

    assert result is model
    assert model.loaded_state == (checkpoint, True)


def test_umt5_checkpoint_can_load_locally_without_world_sync(monkeypatch):
    umt5 = importlib.import_module("rcm.utils.umt5")
    checkpoint = {"encoder.weight": torch.tensor([1.0])}
    events = []

    class FakeModel:
        def __init__(self):
            self.loaded_state = None

        def load_state_dict(self, state, assign):
            self.loaded_state = (state, assign)

    monkeypatch.setattr(umt5.distributed, "is_rank0", lambda: False)
    monkeypatch.setattr(
        torch,
        "load",
        lambda path, map_location: events.append(
            ("load", path, map_location)
        )
        or checkpoint,
    )
    monkeypatch.setattr(
        umt5.distributed,
        "sync_model_states",
        lambda *args, **kwargs: events.append(("sync", args, kwargs)),
    )
    model = FakeModel()

    result = umt5.load_model_torch(
        model,
        "umt5.pth",
        sync_distributed_states=False,
    )

    assert result is model
    assert model.loaded_state == (checkpoint, True)
    assert events == [("load", "umt5.pth", "cpu")]


def test_get_umt5_embedding_uses_npu_by_default(monkeypatch):
    umt5 = importlib.import_module("rcm.utils.umt5")
    events = []

    class FakeEncoder:
        def __init__(self, text_len, device, checkpoint_path):
            events.append(("init", device))

        def __call__(self, prompts, device):
            events.append(("forward", device))
            return "embedding"

    monkeypatch.setattr(umt5, "UMT5EncoderModel", FakeEncoder)
    umt5.t5_encoder = None

    result = umt5.get_umt5_embedding("umt5.pth", "a prompt")

    assert result == "embedding"
    assert events == [("init", "npu"), ("forward", "npu")]


def test_clear_umt5_memory_empties_npu_cache(monkeypatch):
    umt5 = importlib.import_module("rcm.utils.umt5")
    cache_events = []
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(empty_cache=lambda: cache_events.append("npu")),
        raising=False,
    )
    umt5.t5_encoder = object()

    umt5.clear_umt5_memory()

    assert umt5.t5_encoder is None
    assert cache_events == ["npu"]
