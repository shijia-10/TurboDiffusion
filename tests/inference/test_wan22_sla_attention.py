import importlib
import types
import sys

import pytest
import torch


class FakeSparseLinearAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.seen = None

    def forward(self, q, k, v):
        self.seen = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        return q


class AttentionPair(torch.nn.Module):
    def __init__(self, wan2pt2):
        super().__init__()
        self.self_attn = wan2pt2.WanSelfAttention(dim=256, num_heads=2)
        self.cross_attn = wan2pt2.WanCrossAttention(dim=256, num_heads=2)


def test_sla_replaces_only_wan22_self_attention(monkeypatch):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    modify_model = importlib.import_module("modify_model")
    mindie_layers = types.ModuleType("mindiesd.layers")
    mindie_layers.SparseLinearAttention = FakeSparseLinearAttention
    monkeypatch.setitem(sys.modules, "mindiesd.layers", mindie_layers)
    model = AttentionPair(wan2pt2)

    result = modify_model.replace_attention(model, "sla", sla_topk=0.15)

    assert result is model
    assert isinstance(
        model.self_attn.attn_op.local_attn,
        FakeSparseLinearAttention,
    )
    assert model.self_attn.attn_op.local_attn.kwargs == {
        "head_dim": 128,
        "topk": 0.15,
        "BLKQ": 128,
        "BLKK": 128,
    }
    assert model.cross_attn.attn_op.local_attn is wan2pt2.mindie_dense_attention


def test_configured_sla_receives_bnsd_qkv(monkeypatch):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    modify_model = importlib.import_module("modify_model")
    mindie_layers = types.ModuleType("mindiesd.layers")
    mindie_layers.SparseLinearAttention = FakeSparseLinearAttention
    monkeypatch.setitem(sys.modules, "mindiesd.layers", mindie_layers)
    model = AttentionPair(wan2pt2)
    modify_model.replace_attention(model, "sla", sla_topk=0.1)

    output = model.self_attn(
        torch.randn(1, 4, 256),
        seq_lens=torch.tensor([4]),
        freqs=torch.zeros(4, 64),
    )

    assert model.self_attn.attn_op.local_attn.seen == ((1, 2, 4, 128),) * 3
    assert output.shape == (1, 4, 256)


def test_bnsd_ulysses_exchanges_sequence_for_heads_and_restores_local_output(
    monkeypatch,
):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    a2a_cp = importlib.import_module("rcm.utils.a2a_cp")
    local_attention = FakeSparseLinearAttention()
    attention = wan2pt2.WanSelfAttention(
        dim=4,
        num_heads=4,
        qk_norm=False,
    )
    attention.attn_op.local_attn = local_attention
    attention.attn_op.pg = object()
    monkeypatch.setattr(a2a_cp.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        a2a_cp.dist,
        "all_to_all_single",
        lambda output, input, **kwargs: output.copy_(input),
    )
    q = torch.arange(8, dtype=torch.float32).view(1, 4, 2, 1)

    output = attention._apply_attention(q, q, q)

    assert local_attention.seen == ((1, 2, 4, 1),) * 3
    assert output.shape == (1, 2, 4, 1)


def test_profiled_bnsd_ulysses_reports_each_stage_and_communication_ratio(
    monkeypatch,
):
    a2a_cp = importlib.import_module("rcm.utils.a2a_cp")
    clock = [0.0]
    a2a_calls = []
    profiles = []

    def synchronize():
        clock[0] += 1.0

    def fake_a2a(value, scatter_heads, group):
        a2a_calls.append((value, scatter_heads, group))
        return f"a2a({value})"

    monkeypatch.setattr(
        a2a_cp.torch,
        "npu",
        types.SimpleNamespace(synchronize=synchronize),
        raising=False,
    )
    monkeypatch.setattr(a2a_cp.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(a2a_cp, "single_all_to_all_bnsd", fake_a2a)

    output = a2a_cp.bnsd_ulysses_attention(
        "q",
        "k",
        "v",
        lambda q, k, v: f"sla({q},{k},{v})",
        group="ulysses",
        profile_callback=profiles.append,
    )

    assert output == "a2a(sla(a2a(q),a2a(k),a2a(v)))"
    assert a2a_calls == [
        ("q", True, "ulysses"),
        ("k", True, "ulysses"),
        ("v", True, "ulysses"),
        ("sla(a2a(q),a2a(k),a2a(v))", False, "ulysses"),
    ]
    assert profiles == [
        {
            "q_a2a": 1.0,
            "k_a2a": 1.0,
            "v_a2a": 1.0,
            "input_a2a": 3.0,
            "sla": 1.0,
            "output_a2a": 1.0,
            "attention_total": 5.0,
            "communication_ratio": pytest.approx(0.8),
            "hideable_per_block": 1.0,
        }
    ]


def test_attention_profile_selects_only_configured_self_attention_layers(
    monkeypatch,
):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    profile_callbacks = []

    def fake_ulysses(q, k, v, local_attention, group, profile_callback=None):
        profile_callbacks.append(profile_callback)
        return q

    monkeypatch.setattr(wan2pt2, "ATTENTION_PROFILE_LAYERS", {20})
    monkeypatch.setattr(wan2pt2, "bnsd_ulysses_attention", fake_ulysses)

    unprofiled = wan2pt2.WanSelfAttention(dim=4, num_heads=1)
    unprofiled.layer_id = 0
    unprofiled.attn_op.pg = object()
    profiled = wan2pt2.WanSelfAttention(dim=4, num_heads=1)
    profiled.layer_id = 20
    profiled.attn_op.pg = object()
    q = torch.zeros(1, 1, 2, 4)

    unprofiled._apply_attention(q, q, q)
    profiled._apply_attention(q, q, q)

    assert profile_callbacks[0] is None
    assert callable(profile_callbacks[1])


def test_wan_model_assigns_profile_identity_to_each_self_attention():
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    model = wan2pt2.WanModel(
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=20,
        dim=12,
        ffn_dim=24,
        freq_dim=8,
        text_dim=16,
        out_dim=16,
        num_heads=1,
        num_layers=3,
    )

    assert [block.self_attn.layer_id for block in model.blocks] == [0, 1, 2]
    assert [
        block.self_attn.profile_total_layers for block in model.blocks
    ] == [3, 3, 3]


def test_enable_context_parallel_does_not_create_cuda_stream(monkeypatch):
    wan2pt2 = importlib.import_module("rcm.networks.wan2pt2")
    group = object()
    monkeypatch.setattr(
        wan2pt2,
        "get_process_group_ranks",
        lambda process_group: [0, 1],
    )
    monkeypatch.setattr(
        wan2pt2.torch.cuda,
        "Stream",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA stream created")),
    )
    model = wan2pt2.WanModel(
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=8,
        in_dim=20,
        dim=12,
        ffn_dim=24,
        freq_dim=8,
        text_dim=16,
        out_dim=16,
        num_heads=1,
        num_layers=1,
    )

    model.enable_context_parallel(group)

    assert model._cp_group is group
    assert model.blocks[0].self_attn.attn_op.pg is group
    assert model.blocks[0].self_attn.attn_op.stream is None
