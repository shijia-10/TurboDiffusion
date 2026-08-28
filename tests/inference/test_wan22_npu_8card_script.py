import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "inference_wan2.2_i2v_npu_8card.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_eight_card_script_invokes_non_quantized_ulysses_sla(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_args = tmp_path / "torchrun-args.txt"
    capture_env = tmp_path / "torchrun-env.txt"
    _write_executable(
        fake_bin / "torchrun",
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n"
        "printf 'ASCEND_RT_VISIBLE_DEVICES=%s\\nFAST_LAYERNORM=%s\\n' \"$ASCEND_RT_VISIBLE_DEVICES\" \"$FAST_LAYERNORM\" > \"$CAPTURE_ENV\"\n",
    )
    cann_env = tmp_path / "set_env.sh"
    cann_env.write_text("")
    output_path = tmp_path / "output" / "video.mp4"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CANN_ENV": str(cann_env),
            "MODEL_DIR": "/weights/TurboWan2.2-I2V-A14B-720P",
            "IMAGE_PATH": "assets/i2v_inputs/i2v_input_0.jpg",
            "SAVE_PATH": str(output_path),
            "CAPTURE_ARGS": str(capture_args),
            "CAPTURE_ENV": str(capture_env),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    args = capture_args.read_text().splitlines()
    assert args[:2] == [
        "--nproc_per_node=8",
        "turbodiffusion/inference/wan2.2_i2v_infer.py",
    ]
    assert args[args.index("--ulysses-size") + 1] == "8"
    assert args[args.index("--num_frames") + 1] == "81"
    assert args[args.index("--attention_type") + 1] == "sla"
    assert args[args.index("--sla_topk") + 1] == "0.1"
    assert args[args.index("--save_path") + 1] == str(output_path)
    assert args[args.index("--high_noise_model_path") + 1].endswith(
        "TurboWan2.2-I2V-A14B-high-720P.pth"
    )
    assert args[args.index("--low_noise_model_path") + 1].endswith(
        "TurboWan2.2-I2V-A14B-low-720P.pth"
    )
    assert "--default_norm" in args
    assert "--quant_linear" not in args
    assert "--device_id" not in args
    assert output_path.parent.is_dir()
    environment = capture_env.read_text()
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in environment
    assert "FAST_LAYERNORM=1" in environment
