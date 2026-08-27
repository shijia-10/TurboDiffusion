import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "inference_wan2.2_i2v_npu_single.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_single_npu_script_invokes_non_quantized_sla_inference(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for filename in (
        "TurboWan2.2-I2V-A14B-high-720P.pth",
        "TurboWan2.2-I2V-A14B-low-720P.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ):
        (model_dir / filename).touch()

    image_path = tmp_path / "input.jpg"
    image_path.touch()
    cann_env = tmp_path / "set_env.sh"
    cann_env.write_text("")

    capture_args = tmp_path / "python-args.txt"
    capture_env = tmp_path / "python-env.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "python",
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n"
        "printf 'ASCEND_RT_VISIBLE_DEVICES=%s\\nPYTHONPATH=%s\\n' \"$ASCEND_RT_VISIBLE_DEVICES\" \"$PYTHONPATH\" > \"$CAPTURE_ENV\"\n",
    )

    output_path = tmp_path / "output" / "sla.mp4"
    env = os.environ.copy()
    env.update(
        {
            "CANN_ENV": str(cann_env),
            "MODEL_DIR": str(model_dir),
            "IMAGE_PATH": str(image_path),
            "SAVE_PATH": str(output_path),
            "NPU_ID": "3",
            "CAPTURE_ARGS": str(capture_args),
            "CAPTURE_ENV": str(capture_env),
            "PATH": f"{fake_bin}:{env['PATH']}",
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
    assert args[0] == "turbodiffusion/inference/wan2.2_i2v_infer.py"
    assert args[args.index("--attention_type") + 1] == "sla"
    assert args[args.index("--num_frames") + 1] == "81"
    assert args[args.index("--high_noise_model_path") + 1].endswith("high-720P.pth")
    assert args[args.index("--low_noise_model_path") + 1].endswith("low-720P.pth")
    assert "--default_norm" in args
    assert "--quant_linear" not in args
    assert output_path.parent.is_dir()
    assert "ASCEND_RT_VISIBLE_DEVICES=3" in capture_env.read_text()


def test_single_npu_script_rejects_unsupported_attention_type():
    env = os.environ.copy()
    env["ATTENTION_TYPE"] = "sagesla"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "ATTENTION_TYPE must be 'sla' or 'original'" in result.stderr
