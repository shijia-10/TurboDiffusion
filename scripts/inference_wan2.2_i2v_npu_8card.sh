#!/usr/bin/env bash

set -e

export PYTHONPATH=turbodiffusion

source "${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export FAST_LAYERNORM=1

MODEL_DIR="${MODEL_DIR:-/root/weights/TurboWan2.2-I2V-A14B-720P}"
IMAGE_PATH="${IMAGE_PATH:-assets/i2v_inputs/i2v_input_0.jpg}"
SAVE_PATH="${SAVE_PATH:-output/wan2.2_i2v_8card_720p_81f_sla.mp4}"
PROMPT="${PROMPT:-A cinematic video of the subject moving naturally}"

mkdir -p "$(dirname "${SAVE_PATH}")"

torchrun --nproc_per_node=8 turbodiffusion/inference/wan2.2_i2v_infer.py \
    --model Wan2.2-A14B \
    --high_noise_model_path "${MODEL_DIR}/TurboWan2.2-I2V-A14B-high-720P.pth" \
    --low_noise_model_path "${MODEL_DIR}/TurboWan2.2-I2V-A14B-low-720P.pth" \
    --text_encoder_path "${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth" \
    --vae_path "${MODEL_DIR}/Wan2.1_VAE.pth" \
    --resolution 720p \
    --aspect_ratio 16:9 \
    --adaptive_resolution \
    --image_path "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --num_samples 1 \
    --num_steps 4 \
    --num_frames 81 \
    --attention_type sla \
    --sla_topk 0.1 \
    --default_norm \
    --ulysses-size 8 \
    --save_path "${SAVE_PATH}" \
    --ode
