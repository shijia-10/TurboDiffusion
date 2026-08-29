# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os

import torch
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True


def initialize_npu(device_id: int) -> dict:
    """Bind one Ascend NPU and return the tensor placement contract."""
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for Wan2.2 I2V inference") from exc

    if not torch.npu.is_available():
        raise RuntimeError("No available Ascend NPU was detected")

    torch.npu.set_device(device_id)

    return {"device": f"npu:{device_id}", "dtype": torch.bfloat16}


def bind_ulysses_npu(ulysses_size: int) -> tuple[dict, int]:
    """Validate torchrun metadata and bind this rank's local NPU."""
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for Wan2.2 I2V inference") from exc

    values = {}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        try:
            values[name] = int(os.environ[name])
        except KeyError as exc:
            raise ValueError(
                f"Missing {name}; launch multi-NPU inference with torchrun"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc

    rank = values["RANK"]
    local_rank = values["LOCAL_RANK"]
    world_size = values["WORLD_SIZE"]
    if ulysses_size <= 0:
        raise ValueError("ulysses_size must be positive")
    if ulysses_size != world_size:
        raise ValueError(
            f"ulysses_size={ulysses_size} must equal WORLD_SIZE={world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK={rank} must be in [0, {world_size})")
    if not 0 <= local_rank < world_size:
        raise ValueError(
            f"LOCAL_RANK={local_rank} must be in [0, {world_size})"
        )
    if not torch.npu.is_available():
        raise RuntimeError("No available Ascend NPU was detected")

    torch.npu.set_device(local_rank)
    return {"device": f"npu:{local_rank}", "dtype": torch.bfloat16}, rank


def initialize_ulysses_group(rank: int):
    """Initialize HCCL after rank 0 has completed UMT5 inference."""
    log.info(f"[rank={rank}] HCCL initialization begin")
    torch.distributed.init_process_group(backend="hccl", init_method="env://")
    group = torch.distributed.group.WORLD
    log.info(f"[rank={rank}] HCCL initialization end")
    return group


def initialize_inference_npu(args: argparse.Namespace) -> tuple[int, bool]:
    """Bind the selected NPU without starting HCCL before UMT5."""
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc

    if world_size > 1 or args.ulysses_size > 1:
        placement, rank = bind_ulysses_npu(args.ulysses_size)
        use_ulysses = True
    else:
        placement = initialize_npu(args.device_id)
        rank = 0
        use_ulysses = False
    tensor_kwargs.update(placement)
    return rank, use_ulysses

"""Make slight adjustments to H and W so that the final S is evenly divisible by the Ulysses size."""
def align_resolution_for_ulysses(
    height: int,
    width: int,
    latent_frames: int,
    spatial_token_stride: int,
    ulysses_size: int,
) -> tuple[int, int]:
    """Minimally enlarge one dimension so video tokens divide by Ulysses."""
    for name, value in (
        ("height", height),
        ("width", width),
        ("latent_frames", latent_frames),
        ("spatial_token_stride", spatial_token_stride),
        ("ulysses_size", ulysses_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if height % spatial_token_stride or width % spatial_token_stride:
        raise ValueError(
            "height and width must be divisible by spatial_token_stride"
        )

    token_height = height // spatial_token_stride
    token_width = width // spatial_token_stride
    token_count = latent_frames * token_height * token_width
    if token_count % ulysses_size == 0:
        return height, width

    height_multiple = ulysses_size // math.gcd(
        ulysses_size, latent_frames * token_width
    )
    aligned_token_height = (
        (token_height + height_multiple - 1) // height_multiple
    ) * height_multiple

    width_multiple = ulysses_size // math.gcd(
        ulysses_size, latent_frames * token_height
    )
    aligned_token_width = (
        (token_width + width_multiple - 1) // width_multiple
    ) * width_multiple

    height_candidate = aligned_token_height * spatial_token_stride
    width_candidate = aligned_token_width * spatial_token_stride
    if height_candidate * width <= height * width_candidate:
        return height_candidate, width
    return height, width_candidate


def enable_ulysses(models, ulysses_group) -> None:
    """Attach the same Ulysses process group to every noise model."""
    if ulysses_group is None:
        return
    for model in models:
        model.enable_context_parallel(ulysses_group)


def prepare_text_embedding(
    checkpoint_path: str,
    prompt: str,
    sync_distributed_states: bool = True,
) -> torch.Tensor:
    """Compute one text embedding on the already-bound local NPU."""
    log.info("[rank=0] UMT5 embedding begin")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=checkpoint_path,
            prompts=prompt,
            device=tensor_kwargs["device"],
            sync_distributed_states=sync_distributed_states,
        ).to(**tensor_kwargs)
    torch.npu.synchronize()
    clear_umt5_memory()
    log.info("[rank=0] UMT5 embedding end")
    return text_emb


def broadcast_text_embedding(text_emb, rank: int, ulysses_group) -> torch.Tensor:
    """Broadcast the rank-0 UMT5 result after HCCL is initialized."""
    log.info(f"[rank={rank}] text embedding broadcast begin")
    if rank == 0:
        shape = torch.tensor(
            text_emb.shape,
            dtype=torch.long,
            device=tensor_kwargs["device"],
        )
    else:
        text_emb = None
        shape = torch.empty(
            3,
            dtype=torch.long,
            device=tensor_kwargs["device"],
        )

    torch.distributed.broadcast(shape, src=0, group=ulysses_group)
    if rank != 0:
        text_emb = torch.empty(
            tuple(shape.tolist()),
            **tensor_kwargs,
        )
    torch.distributed.broadcast(text_emb, src=0, group=ulysses_group)
    log.info(f"[rank={rank}] text embedding broadcast end")
    return text_emb


def prepare_parallel_text_embedding(
    checkpoint_path: str,
    prompt: str,
    rank: int,
) -> tuple[torch.Tensor, object]:
    """Run rank-0 UMT5 before HCCL, then distribute its embedding."""
    text_emb = None
    if rank == 0:
        text_emb = prepare_text_embedding(
            checkpoint_path=checkpoint_path,
            prompt=prompt,
            sync_distributed_states=False,
        )
    ulysses_group = initialize_ulysses_group(rank)
    text_emb = broadcast_text_embedding(text_emb, rank, ulysses_group)
    return text_emb, ulysses_group


def sampling_progress(timesteps, rank: int):
    """Show the sampling progress bar on rank 0 only."""
    steps = list(zip(timesteps[:-1], timesteps[1:]))
    return tqdm(
        steps,
        desc="Sampling",
        total=len(steps),
        disable=rank != 0,
    )


def decode_and_save_rank0(rank, tokenizer, samples, save_path: str) -> bool:
    """Decode and save the generated video on rank 0 only."""
    if rank != 0:
        return False
    with torch.no_grad():
        video = tokenizer.decode(samples)
    videos = (1.0 + video.float().cpu().unsqueeze(0).clamp(-1, 1)) / 2.0
    save_image_or_video(
        rearrange(videos, "n b c t h w -> c t (n h) (b w)"),
        save_path,
        fps=16,
    )
    return True


def finalize_ulysses(ulysses_group) -> None:
    """Synchronize all Ulysses ranks before releasing the HCCL group."""
    if ulysses_group is None:
        return
    torch.distributed.barrier(group=ulysses_group)
    torch.distributed.destroy_process_group(ulysses_group)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion inference script for Wan2.2 I2V with High/Low Noise models")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image (required unless --serve)")
    parser.add_argument("--high_noise_model_path", type=str, required=True, help="Path to the high-noise model")
    parser.add_argument("--low_noise_model_path", type=str, required=True, help="Path to the low-noise model")
    parser.add_argument("--boundary", type=float, default=0.9, help="Timestep boundary for switching from high to low noise model")
    parser.add_argument("--model", choices=["Wan2.2-A14B"], default="Wan2.2-A14B", help="Model to use")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=200, help="Initial sigma for rCM")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth", help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="Path to the umT5 text encoder")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation (required unless --serve)")
    parser.add_argument("--resolution", default="720p", type=str, help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true", help="If set, adapts the output resolution to the input image's aspect ratio, using the area defined by --resolution and --aspect_ratio as a target.")
    parser.add_argument("--ode", action="store_true", help="Use ODE for sampling (sharper but less robust than SDE)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4", help="Path to save the generated video (include file extension)")
    parser.add_argument("--attention_type", choices=["sla", "original"], default="sla", help="Type of attention mechanism to use")
    parser.add_argument("--sla_topk", type=float, default=0.1, help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true", help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true", help="Whether to replace LayerNorm/RMSNorm layers with faster versions")
    parser.add_argument("--serve", action="store_true", help="Launch interactive TUI server mode (keeps model loaded)")
    parser.add_argument("--device_id", type=int, default=0, help="Ascend NPU device ID")
    parser.add_argument(
        "--ulysses-size",
        type=int,
        default=1,
        help="Ulysses sequence-parallel degree; must equal torchrun WORLD_SIZE",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    rank, use_ulysses = initialize_inference_npu(args)
    ulysses_group = None

    # Handle serve mode
    if args.serve:
        # Set mode to i2v for the TUI server
        args.mode = "i2v"
        from serve.tui import main as serve_main
        serve_main(args)
        exit(0)

    # Validate required args for one-shot mode
    if args.prompt is None:
        log.error("--prompt is required (unless using --serve mode)")
        exit(1)
    if args.image_path is None:
        log.error("--image_path is required (unless using --serve mode)")
        exit(1)

    if use_ulysses:
        text_emb, ulysses_group = prepare_parallel_text_embedding(
            checkpoint_path=args.text_encoder_path,
            prompt=args.prompt,
            rank=rank,
        )
    else:
        text_emb = prepare_text_embedding(
            checkpoint_path=args.text_encoder_path,
            prompt=args.prompt,
        )

    log.info(f"Loading DiT models.")
    high_noise_model = create_model(dit_path=args.high_noise_model_path, args=args).cpu()
    torch.npu.empty_cache()
    low_noise_model = create_model(dit_path=args.low_noise_model_path, args=args).cpu()
    torch.npu.empty_cache()
    enable_ulysses(
        [high_noise_model, low_noise_model],
        ulysses_group,
    )
    log.success(f"Successfully loaded DiT model.")

    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    log.info(f"Loading and preprocessing image from: {args.image_path}")
    input_image = Image.open(args.image_path).convert("RGB")
    if args.adaptive_resolution:
        log.info("Adaptive resolution mode enabled.")
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        log.info(f"Target area is based on {args.resolution} {args.aspect_ratio} (~{max_resolution_area} pixels).")

        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w

        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)

        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride

        log.info(f"Input image aspect ratio: {image_aspect_ratio:.4f}. Adaptive resolution set to: {w}x{h}")
    else:
        log.info("Fixed resolution mode.")
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        log.info(f"Resolution set to: {w}x{h}")
    F = args.num_frames
    spatial_token_stride = tokenizer.spatial_compression_factor * 2
    lat_t = tokenizer.get_latent_num_frames(F)
    if ulysses_group is not None:
        original_h, original_w = h, w
        h, w = align_resolution_for_ulysses(
            height=h,
            width=w,
            latent_frames=lat_t,
            spatial_token_stride=spatial_token_stride,
            ulysses_size=args.ulysses_size,
        )
        if rank == 0 and (h, w) != (original_h, original_w):
            log.info(
                f"Adjusted resolution for Ulysses: "
                f"{original_w}x{original_h} -> {w}x{h}; "
                f"tokens={lat_t}x{h // spatial_token_stride}x"
                f"{w // spatial_token_stride}"
            )
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor

    log.info(f"Preprocessing image to {w}x{h}...")
    image_transforms = T.Compose(
        [
            T.ToImage(),
            T.Resize(size=(h, w), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = image_transforms(input_image).unsqueeze(0).to(device=tensor_kwargs["device"], dtype=torch.float32)

    with torch.no_grad():
        frames_to_encode = torch.cat(
            [image_tensor.unsqueeze(2), torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], dim=2
        )  # -> B, C, T, H, W
        encoded_latents = tokenizer.encode(frames_to_encode)  # -> B, C_lat, T_lat, H_lat, W_lat
        
        del frames_to_encode
        torch.npu.empty_cache()

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, device=tensor_kwargs["device"], dtype=tensor_kwargs["dtype"])
    msk[:, :, 0, :, :] = 1.0

    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    log.info(f"Generating with prompt: {args.prompt}")
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples), "y_B_C_T_H_W": y}

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    high_noise_model.to(tensor_kwargs["device"])
    net = high_noise_model
    switched = False
    for t_cur, t_next in sampling_progress(t_steps, rank):
        if t_cur.item() < args.boundary and not switched:
            high_noise_model.cpu()
            torch.npu.empty_cache()
            low_noise_model.to(tensor_kwargs["device"])
            net = low_noise_model
            switched = True
            log.info("Switched to low noise model.")
        with torch.no_grad():
            v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), **condition).to(
                torch.float64
            )
            if args.ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )
    samples = x.float()
    low_noise_model.cpu()
    torch.npu.empty_cache()

    if decode_and_save_rank0(
        rank,
        tokenizer,
        samples,
        args.save_path,
    ):
        log.success(f"Saved generated video to: {args.save_path}")
    finalize_ulysses(ulysses_group)
