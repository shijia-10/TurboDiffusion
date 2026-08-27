import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


MODIFY_MODEL_PATH = (
    Path(__file__).parents[2]
    / "turbodiffusion/inference/modify_model.py"
)


def test_modify_model_import_does_not_require_optional_cuda_backends(monkeypatch):
    monkeypatch.setitem(sys.modules, "ops", None)
    monkeypatch.setitem(sys.modules, "SLA", None)
    spec = importlib.util.spec_from_file_location(
        "wan22_modify_model_without_cuda_backends",
        MODIFY_MODEL_PATH,
    )
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module.create_model)


def test_mindie_dense_attention_declares_bnsd_layout(monkeypatch):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    calls = []

    def fake_attention_forward(q, k, v, head_first):
        calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), head_first))
        return q

    monkeypatch.setitem(
        sys.modules,
        "mindiesd",
        types.SimpleNamespace(attention_forward=fake_attention_forward),
    )
    q = torch.randn(1, 2, 4, 128)

    output = wan2pt2.mindie_dense_attention(q, q, q)

    assert output is q
    assert calls == [((1, 2, 4, 128),) * 3 + (True,)]


@pytest.mark.parametrize("attention_class", ["WanSelfAttention", "WanCrossAttention"])
def test_original_wan_attention_uses_mindie_dense_backend(
    monkeypatch,
    attention_class,
):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    calls = []

    def fake_attention_forward(q, k, v, head_first):
        calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), head_first))
        return q

    monkeypatch.setitem(
        sys.modules,
        "mindiesd",
        types.SimpleNamespace(attention_forward=fake_attention_forward),
    )
    attention = getattr(wan2pt2, attention_class)(dim=256, num_heads=2)

    if attention_class == "WanSelfAttention":
        output = attention(
            torch.randn(1, 4, 256),
            seq_lens=torch.tensor([4]),
            freqs=torch.zeros(4, 64),
        )
        expected_shapes = ((1, 2, 4, 128),) * 3
    else:
        output = attention(
            torch.randn(1, 4, 256),
            context=torch.randn(1, 6, 256),
            context_lens=None,
        )
        expected_shapes = (
            (1, 2, 4, 128),
            (1, 2, 6, 128),
            (1, 2, 6, 128),
        )

    assert calls == [expected_shapes + (True,)]
    assert output.shape == (1, 4, 256)
