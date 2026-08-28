#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
MODEL_DIR="${MODEL_DIR:-/root/weights/TurboWan2.2-I2V-A14B-720P}"
IMAGE_PATH="${IMAGE_PATH:-assets/i2v_inputs/i2v_input_0.jpg}"
NPU_ID="${NPU_ID:-0}"
ATTENTION_TYPE="${ATTENTION_TYPE:-sla}"
SLA_TOPK="${SLA_TOPK:-0.1}"
PROMPT="${PROMPT:-A cinematic video of the subject moving naturally}"
SAVE_PATH="${SAVE_PATH:-output/wan2.2_i2v_720p_81f_${ATTENTION_TYPE}.mp4}"

HIGH_NOISE_MODEL_PATH="${HIGH_NOISE_MODEL_PATH:-${MODEL_DIR}/TurboWan2.2-I2V-A14B-high-720P.pth}"
LOW_NOISE_MODEL_PATH="${LOW_NOISE_MODEL_PATH:-${MODEL_DIR}/TurboWan2.2-I2V-A14B-low-720P.pth}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth}"
VAE_PATH="${VAE_PATH:-${MODEL_DIR}/Wan2.1_VAE.pth}"

if [[ "${ATTENTION_TYPE}" != "sla" && "${ATTENTION_TYPE}" != "original" ]]; then
    echo "ATTENTION_TYPE must be 'sla' or 'original', got '${ATTENTION_TYPE}'" >&2
    exit 2
fi

require_file() {
    local path="$1"
    local description="$2"
    if [[ ! -f "${path}" ]]; then
        echo "Missing ${description}: ${path}" >&2
        exit 1
    fi
}

require_file "${CANN_ENV}" "CANN environment script"
require_file "${HIGH_NOISE_MODEL_PATH}" "high-noise checkpoint"
require_file "${LOW_NOISE_MODEL_PATH}" "low-noise checkpoint"
require_file "${TEXT_ENCODER_PATH}" "text encoder checkpoint"
require_file "${VAE_PATH}" "VAE checkpoint"
require_file "${IMAGE_PATH}" "input image"

# shellcheck disable=SC1090
source "${CANN_ENV}"

export PYTHONPATH="${REPO_ROOT}/turbodiffusion${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export FAST_LAYERNORM="${FAST_LAYERNORM:-1}"

mkdir -p "$(dirname "${SAVE_PATH}")"

echo "Running Wan2.2 I2V single-NPU inference"
echo "  physical NPU: ${NPU_ID}"
echo "  attention:    ${ATTENTION_TYPE}"
echo "  fast norm:    ${FAST_LAYERNORM}"
echo "  input:        ${IMAGE_PATH}"
echo "  output:       ${SAVE_PATH}"

# ASCEND_RT_VISIBLE_DEVICES maps the selected physical NPU to logical npu:0.
python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --device_id 0 \
    --model Wan2.2-A14B \
    --high_noise_model_path "${HIGH_NOISE_MODEL_PATH}" \
    --low_noise_model_path "${LOW_NOISE_MODEL_PATH}" \
    --text_encoder_path "${TEXT_ENCODER_PATH}" \
    --vae_path "${VAE_PATH}" \
    --resolution 720p \
    --aspect_ratio 16:9 \
    --adaptive_resolution \
    --image_path "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --num_samples 1 \
    --num_steps 4 \
    --num_frames 81 \
    --attention_type "${ATTENTION_TYPE}" \
    --sla_topk "${SLA_TOPK}" \
    --default_norm \
    --save_path "${SAVE_PATH}" \
    --ode
