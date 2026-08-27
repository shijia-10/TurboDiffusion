import ast
import importlib
import math
from pathlib import Path

import pytest
import torch


NETWORK_PATH = (
    Path(__file__).parents[2]
    / "turbodiffusion/rcm/networks/wan2pt2.py"
)


class FakeRangeTensor:
    def __init__(self, npu_events):
        self.npu_events = npu_events

    def float(self):
        return self

    def __getitem__(self, key):
        return self

    def __truediv__(self, value):
        return self

    def npu(self):
        self.npu_events.append("npu")
        return self


class CaptureLocalAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, q, k, v):
        self.seen = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        return q


def test_rope_cache_parameters_are_created_on_npu(monkeypatch):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    npu_events = []
    monkeypatch.setattr(
        wan2pt2.torch,
        "arange",
        lambda *args, **kwargs: FakeRangeTensor(npu_events),
    )
    rope = wan2pt2.VideoRopePosition3DEmb(
        head_dim=128,
        len_h=16,
        len_w=16,
        len_t=16,
    )

    rope.cache_parameters()

    assert npu_events == ["npu", "npu", "npu"]


def test_all_dit_autocast_regions_select_npu():
    tree = ast.parse(NETWORK_PATH.read_text())
    autocast_devices = [
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "amp"
        and call.func.attr == "autocast"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    ]

    assert len(autocast_devices) == 6
    assert set(autocast_devices) == {"npu"}


def test_rope_apply_uses_bnsd_layout():
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    bnsd = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]
    )
    freqs = torch.tensor([[math.pi / 2, 0.0], [0.0, 0.0]])

    actual = wan2pt2.rope_apply(bnsd, freqs)

    expected = torch.tensor(
        [[[[-2.0, 1.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("attention_class", ["WanSelfAttention", "WanCrossAttention"])
def test_wan_attention_bnsd_calls_local_backend_and_restores_bsc(attention_class):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    attention = getattr(wan2pt2, attention_class)(dim=256, num_heads=2)
    backend = CaptureLocalAttention()
    attention.attn_op.local_attn = backend

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

    assert backend.seen == expected_shapes
    assert output.shape == (1, 4, 256)


def test_bnsd_attention_rejects_context_parallel_before_local_backend():
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    attention = wan2pt2.WanSelfAttention(dim=256, num_heads=2)
    backend = CaptureLocalAttention()
    attention.attn_op.local_attn = backend
    attention.attn_op.pg = object()

    with pytest.raises(
        RuntimeError,
        match="BNSD Wan attention does not support context parallel in phase 1",
    ):
        attention(
            torch.randn(1, 4, 256),
            seq_lens=torch.tensor([4]),
            freqs=torch.zeros(4, 64),
        )

    assert backend.seen is None
