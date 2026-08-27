import importlib
import types
import sys

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
