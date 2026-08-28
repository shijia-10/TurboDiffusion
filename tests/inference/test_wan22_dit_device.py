import ast
from contextlib import nullcontext
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


class ReturnZeros(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        del args, kwargs
        return torch.zeros_like(x)


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


def test_wan_rms_norm_preserves_bfloat16_activation_dtype():
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    norm = wan2pt2.WanRMSNorm(4)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)

    output = norm(x)

    assert norm.weight.dtype == torch.float32
    assert output.dtype == x.dtype


@pytest.mark.parametrize("enabled, expected_call_count", [(0, 0), (1, 3)])
def test_fast_layernorm_switch_controls_attention_block_norms(
    monkeypatch,
    enabled,
    expected_call_count,
):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    calls = []

    def fake_fast_layernorm(norm, x):
        calls.append(norm)
        return norm(x)

    monkeypatch.setattr(wan2pt2, "FAST_LAYERNORM", enabled, raising=False)
    monkeypatch.setattr(
        wan2pt2,
        "fast_layernorm",
        fake_fast_layernorm,
        raising=False,
    )
    monkeypatch.setattr(
        wan2pt2.amp,
        "autocast",
        lambda *args, **kwargs: nullcontext(),
    )
    block = wan2pt2.WanAttentionBlock(
        "i2v_cross_attn",
        dim=4,
        ffn_dim=8,
        num_heads=1,
        cross_attn_norm=True,
    )
    block.self_attn = ReturnZeros()
    block.cross_attn = ReturnZeros()
    block.ffn = ReturnZeros()

    output = block(
        x=torch.randn(1, 2, 4),
        e=torch.zeros(1, 6, 4),
        seq_lens=torch.tensor([2]),
        freqs=torch.zeros(2, 2),
        context=torch.randn(1, 3, 4),
        context_lens=None,
    )

    assert output.shape == (1, 2, 4)
    assert len(calls) == expected_call_count
    if enabled:
        assert calls == [block.norm1, block.norm3, block.norm2]


def test_context_parallel_split_preserves_current_tensor_device(monkeypatch):
    context_parallel = importlib.import_module("rcm.utils.context_parallel")

    class Group:
        @staticmethod
        def rank():
            return 1

    monkeypatch.setattr(
        context_parallel,
        "get_process_group_ranks",
        lambda group: [0, 1],
    )
    x = torch.arange(8).view(1, 8, 1)

    output = context_parallel.split_inputs_cp(
        x,
        seq_dim=1,
        cp_group=Group(),
    )

    assert output.device == x.device
    torch.testing.assert_close(output.flatten(), torch.tensor([4, 5, 6, 7]))


def test_context_parallel_broadcast_preserves_current_tensor_device(
    monkeypatch,
):
    context_parallel = importlib.import_module("rcm.utils.context_parallel")
    group = object()
    monkeypatch.setattr(
        context_parallel,
        "get_process_group_ranks",
        lambda process_group: [0, 1],
    )
    monkeypatch.setattr(
        context_parallel.distributed,
        "get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        context_parallel.torch.distributed,
        "broadcast",
        lambda tensor, src, group: None,
    )
    x = torch.arange(4).view(1, 4)

    output = context_parallel.broadcast(x, group)

    assert output.device == x.device
    torch.testing.assert_close(output, x)


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
