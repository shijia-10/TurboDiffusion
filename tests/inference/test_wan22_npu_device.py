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
    utils_module.log = types.SimpleNamespace(info=lambda *args, **kwargs: None)
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


def test_i2v_cli_exposes_only_ulysses_parallel_degree(
    inference_module,
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wan2.2_i2v_infer.py",
            "--high_noise_model_path",
            "high.pth",
            "--low_noise_model_path",
            "low.pth",
        ],
    )

    args = inference_module.parse_arguments()

    assert args.ulysses_size == 1
    assert not hasattr(args, "tensor_parallel_size")
    assert not hasattr(args, "distributed_backend")


def test_bind_ulysses_npu_binds_local_rank_without_initializing_hccl(
    inference_module,
    monkeypatch,
):
    events = []
    fake_npu = types.SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda device_id: events.append(("set_device", device_id)),
    )
    fake_dist = types.SimpleNamespace(
        init_process_group=lambda **kwargs: events.append(
            ("init_process_group", kwargs)
        ),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "distributed", fake_dist)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    tensor_config, rank = inference_module.bind_ulysses_npu(8)

    assert tensor_config == {"device": "npu:3", "dtype": torch.bfloat16}
    assert rank == 3
    assert events == [("set_device", 3)]


def test_initialize_ulysses_group_initializes_hccl_after_device_binding(
    inference_module,
    monkeypatch,
):
    events = []
    world_group = object()
    fake_dist = types.SimpleNamespace(
        group=types.SimpleNamespace(WORLD=world_group),
        init_process_group=lambda **kwargs: events.append(
            ("init_process_group", kwargs)
        ),
    )
    monkeypatch.setattr(torch, "distributed", fake_dist)

    group = inference_module.initialize_ulysses_group(rank=3)

    assert group is world_group
    assert events == [
        (
            "init_process_group",
            {"backend": "hccl", "init_method": "env://"},
        )
    ]


def test_bind_ulysses_npu_rejects_world_size_mismatch_before_npu_setup(
    inference_module,
    monkeypatch,
):
    events = []
    fake_npu = types.SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda device_id: events.append(("set_device", device_id)),
    )
    fake_dist = types.SimpleNamespace(
        init_process_group=lambda **kwargs: events.append("init_process_group")
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "distributed", fake_dist)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")

    with pytest.raises(ValueError, match="ulysses_size.*WORLD_SIZE"):
        inference_module.bind_ulysses_npu(4)

    assert events == []


def test_initialize_inference_npu_selects_ulysses_for_torchrun_world(
    inference_module,
    monkeypatch,
):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(
        inference_module,
        "bind_ulysses_npu",
        lambda size: (
            {"device": "npu:6", "dtype": torch.bfloat16},
            6,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        inference_module,
        "initialize_npu",
        lambda device_id: pytest.fail("single-NPU initializer was used"),
    )
    inference_module.tensor_kwargs.clear()

    rank, use_ulysses = inference_module.initialize_inference_npu(
        types.SimpleNamespace(ulysses_size=8, device_id=0)
    )

    assert rank == 6
    assert use_ulysses is True
    assert inference_module.tensor_kwargs == {
        "device": "npu:6",
        "dtype": torch.bfloat16,
    }


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (1104, 832, (1120, 832)),
        (832, 1104, (832, 1120)),
    ],
)
def test_align_resolution_for_eight_way_ulysses_uses_smallest_area_growth(
    inference_module,
    height,
    width,
    expected,
):
    aligned = inference_module.align_resolution_for_ulysses(
        height=height,
        width=width,
        latent_frames=21,
        spatial_token_stride=16,
        ulysses_size=8,
    )

    assert aligned == expected
    aligned_height, aligned_width = aligned
    token_count = (
        21 * (aligned_height // 16) * (aligned_width // 16)
    )
    assert token_count % 8 == 0


def test_align_resolution_keeps_already_divisible_720p_shape(inference_module):
    assert inference_module.align_resolution_for_ulysses(
        height=720,
        width=1280,
        latent_frames=21,
        spatial_token_stride=16,
        ulysses_size=8,
    ) == (720, 1280)


def test_enable_ulysses_sets_same_group_on_both_noise_models(inference_module):
    events = []

    class Model:
        def __init__(self, name):
            self.name = name

        def enable_context_parallel(self, group):
            events.append((self.name, group))

    group = object()
    high = Model("high")
    low = Model("low")

    inference_module.enable_ulysses([high, low], group)

    assert events == [("high", group), ("low", group)]


@pytest.mark.parametrize("rank", [0, 7])
def test_every_rank_initializes_hccl_before_computing_umt5_locally(
    inference_module,
    monkeypatch,
    rank,
):
    group = object()
    expected = torch.arange(24, dtype=torch.bfloat16).view(1, 3, 8)
    events = []

    monkeypatch.setattr(
        inference_module,
        "get_umt5_embedding",
        lambda **kwargs: events.append(("compute", kwargs)) or expected,
    )
    monkeypatch.setattr(
        inference_module,
        "clear_umt5_memory",
        lambda: events.append(("clear",)),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            synchronize=lambda: events.append(("synchronize",))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        inference_module,
        "initialize_ulysses_group",
        lambda rank: events.append(("init_hccl", rank)) or group,
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda *args, **kwargs: pytest.fail(
            "symmetric UMT5 execution must not broadcast text embeddings"
        ),
    )
    inference_module.tensor_kwargs.clear()
    inference_module.tensor_kwargs.update(
        {"device": "cpu", "dtype": torch.bfloat16}
    )

    actual, actual_group = inference_module.prepare_parallel_text_embedding(
        checkpoint_path="umt5.pth",
        prompt="a cat",
        rank=rank,
    )

    torch.testing.assert_close(actual, expected)
    assert actual_group is group
    assert events == [
        ("init_hccl", rank),
        (
            "compute",
            {
                "checkpoint_path": "umt5.pth",
                "prompts": "a cat",
                "device": "cpu",
                "sync_distributed_states": False,
            },
        ),
        ("synchronize",),
        ("clear",),
    ]


def test_nonzero_rank_skips_video_decode_and_save(inference_module):
    class Tokenizer:
        def decode(self, samples):
            raise AssertionError("nonzero rank decoded the video")

    saved = inference_module.decode_and_save_rank0(
        rank=3,
        tokenizer=Tokenizer(),
        samples=torch.zeros(1, 16, 2, 2, 2),
        save_path="output.mp4",
    )

    assert saved is False


def test_rank0_decodes_and_saves_video(inference_module, monkeypatch):
    events = []

    class Tokenizer:
        def decode(self, samples):
            events.append(("decode", tuple(samples.shape)))
            return torch.zeros(1, 3, 2, 2, 2)

    monkeypatch.setattr(
        inference_module,
        "save_image_or_video",
        lambda video, path, fps: events.append(
            ("save", tuple(video.shape), path, fps)
        ),
    )

    saved = inference_module.decode_and_save_rank0(
        rank=0,
        tokenizer=Tokenizer(),
        samples=torch.zeros(1, 16, 2, 2, 2),
        save_path="output.mp4",
    )

    assert saved is True
    assert events == [
        ("decode", (1, 16, 2, 2, 2)),
        ("save", (3, 2, 2, 2), "output.mp4", 16),
    ]


def test_sampling_progress_is_hidden_on_nonzero_rank(
    inference_module,
    monkeypatch,
):
    seen = []
    monkeypatch.setattr(
        inference_module,
        "tqdm",
        lambda values, **kwargs: seen.append(kwargs) or values,
    )
    timesteps = torch.tensor([1.0, 0.5, 0.0])

    list(inference_module.sampling_progress(timesteps, rank=4))

    assert seen == [{"desc": "Sampling", "total": 2, "disable": True}]


def test_finalize_ulysses_barriers_before_destroy(inference_module, monkeypatch):
    events = []
    group = object()
    fake_dist = types.SimpleNamespace(
        barrier=lambda **kwargs: events.append(("barrier", kwargs)),
        destroy_process_group=lambda process_group: events.append(
            ("destroy", process_group)
        ),
    )
    monkeypatch.setattr(torch, "distributed", fake_dist)

    inference_module.finalize_ulysses(group)

    assert events == [
        ("barrier", {"group": group}),
        ("destroy", group),
    ]
