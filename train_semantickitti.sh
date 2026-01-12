#!/usr/bin/env bash

# RangeLDM training on SemanticKITTI
# Fully rewritten: supports training from scratch (RangeDM, no VAE) or with VAE (RangeLDM).
#
# Usage examples:
#   # 1) Train from scratch (no VAE) on 1 GPU
#   ./train_semantickitti.sh --mode rangedm --gpus 1 --mp fp16
#
#   # 2) Train LDM (with VAE). You must provide a VAE checkpoint path.
#   ./train_semantickitti.sh --mode ldm --vae-checkpoint /path/to/vae.ckpt --gpus 2 --mp bf16
#
#   # 3) Custom dataset root
#   ./train_semantickitti.sh --dataset /data/SemanticKITTI --mode rangedm

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_CONFIG=""

print_help() {
    cat <<EOF
RangeLDM training on SemanticKITTI

Flags:
    --mode {rangedm|ldm}       Training mode. 'rangedm' (no VAE) is recommended to start. Default: rangedm
    --dataset PATH             SemanticKITTI root (or .../dataset or .../sequences). Default: auto-detect
    --config PATH              Override config file. If omitted, uses sensible defaults per mode.
    --vae-checkpoint PATH      Required for --mode ldm; inserted into a temp copy of the config.
    --vae-config PATH          Optional VAE config path to insert into the temp config (ldm mode).
    --gpus N                   Number of GPUs (accelerate multi-process). Default: 1
    --mp {no|fp16|bf16}        Mixed precision. Default: no
    --out DIR                  Optional output_dir override written into the temp config.
    --help                     Show this help and exit.

Environment:
    SEMANTICKITTI_DATASET      If set, used as dataset root unless --dataset is provided.

Examples:
    ./train_semantickitti.sh --mode rangedm --gpus 1 --mp fp16
    ./train_semantickit ti.sh --mode ldm --vae-checkpoint /checkpoints/vae.ckpt --gpus 2
EOF
}

# Defaults
MODE="rangedm"           # 'rangedm' (no VAE) or 'ldm' (with VAE)
DATASET_ROOT="${SEMANTICKITTI_DATASET:-}"
USER_CONFIG=""
VAE_CHECKPOINT=""
VAE_CONFIG_OVERRIDE=""
NUM_GPUS=1
MIXED_PRECISION="no"
OUTPUT_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2;;
        --dataset) DATASET_ROOT="$2"; shift 2;;
        --config) USER_CONFIG="$2"; shift 2;;
        --vae-checkpoint) VAE_CHECKPOINT="$2"; shift 2;;
        --vae-config) VAE_CONFIG_OVERRIDE="$2"; shift 2;;
        --gpus) NUM_GPUS="$2"; shift 2;;
        --mp) MIXED_PRECISION="$2"; shift 2;;
        --out) OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
        --help|-h) print_help; exit 0;;
        *) echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

# Auto-detect dataset root if not set
if [[ -z "${DATASET_ROOT}" ]]; then
    if [[ -d "/datasets/SemanticKITTI" ]]; then
        DATASET_ROOT="/datasets/SemanticKITTI"
    elif [[ -d "${ROOT_DIR}/../container-3d-diffusion-lidar-super-resolution/lidiff/Datasets/SemanticKITTI" ]]; then
        DATASET_ROOT="${ROOT_DIR}/../container-3d-diffusion-lidar-super-resolution/lidiff/Datasets/SemanticKITTI"
    else
        echo "ERROR: Could not auto-detect SemanticKITTI. Provide --dataset or set SEMANTICKITTI_DATASET." >&2
        exit 1
    fi
fi

# Resolve sequences path
resolve_sequences_path() {
    local base="$1"
    if [[ -d "${base}/dataset/sequences" ]]; then
        echo "${base}/dataset/sequences"
    elif [[ -d "${base}/sequences" ]]; then
        echo "${base}/sequences"
    else
        echo "${base}"
    fi
}

SEQUENCES_PATH="$(resolve_sequences_path "${DATASET_ROOT}")"
if [[ ! -d "${SEQUENCES_PATH}" ]]; then
    echo "ERROR: Cannot find sequences directory in ${DATASET_ROOT}" >&2
    exit 1
fi

echo "=========================================="
echo "RangeLDM Training on SemanticKITTI"
echo "=========================================="
echo "Repo root        : ${ROOT_DIR}"
echo "Dataset root     : ${DATASET_ROOT}"
echo "Sequences path   : ${SEQUENCES_PATH}"
echo "Mode             : ${MODE}"
echo "GPUs             : ${NUM_GPUS}"
echo "Mixed precision  : ${MIXED_PRECISION}"
echo "=========================================="
echo

# Quick dataset check (train sequences)
for seq in 00 01 02 03 04 05 06 07 09 10; do
    d="${SEQUENCES_PATH}/${seq}/velodyne"
    if [[ -d "$d" ]]; then
        n=$(ls -1 "$d"/*.bin 2>/dev/null | wc -l || true)
        printf "  seq %s: %s scans\n" "$seq" "$n"
    else
        printf "  seq %s: MISSING\n" "$seq"
    fi
done
echo

# Select config
if [[ -n "${USER_CONFIG}" ]]; then
    BASE_CONFIG="${USER_CONFIG}"
else
    if [[ "${MODE}" == "ldm" ]]; then
        BASE_CONFIG="${ROOT_DIR}/ldm/configs/semantickitti.yaml"
    else
        BASE_CONFIG="${ROOT_DIR}/ldm/configs/semantickitti_rangedm.yaml"
    fi
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "ERROR: Config not found: ${BASE_CONFIG}" >&2
    exit 1
fi

# Prepare config (for LDM insert VAE checkpoint and optional overrides)
TRAIN_CONFIG="${BASE_CONFIG}"
cleanup() {
    [[ -n "${TMP_CONFIG}" && -f "${TMP_CONFIG}" ]] && rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

if [[ "${MODE}" == "ldm" ]]; then
    if [[ -z "${VAE_CHECKPOINT}" ]]; then
        echo "ERROR: --mode ldm requires --vae-checkpoint PATH" >&2
        exit 1
    fi
    if [[ ! -f "${VAE_CHECKPOINT}" ]]; then
        echo "ERROR: VAE checkpoint not found: ${VAE_CHECKPOINT}" >&2
        exit 1
    fi
    TMP_CONFIG="$(mktemp "${ROOT_DIR}/semkitti_ldm_cfg.XXXXXX.yaml")"
    cp "${BASE_CONFIG}" "${TMP_CONFIG}"
    # Insert checkpoint path (replace the line starting with vae_checkpoint:)
    sed -i -E "s|^vae_checkpoint:.*$|vae_checkpoint: ${VAE_CHECKPOINT//\/\\}|" "${TMP_CONFIG}"
    # Optional: override vae_config and/or output_dir
    if [[ -n "${VAE_CONFIG_OVERRIDE}" ]]; then
        sed -i -E "s|^vae_config:.*$|vae_config: ${VAE_CONFIG_OVERRIDE//\/\\}|" "${TMP_CONFIG}"
    fi
    if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
        sed -i -E "s|^output_dir:.*$|output_dir: ${OUTPUT_DIR_OVERRIDE//\/\\}|" "${TMP_CONFIG}"
    fi
    TRAIN_CONFIG="${TMP_CONFIG}"
else
    # RangeDM: optional output_dir override
    if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
        TMP_CONFIG="$(mktemp "${ROOT_DIR}/semkitti_rangedm_cfg.XXXXXX.yaml")"
        cp "${BASE_CONFIG}" "${TMP_CONFIG}"
        sed -i -E "s|^output_dir:.*$|output_dir: ${OUTPUT_DIR_OVERRIDE//\/\\}|" "${TMP_CONFIG}"
        TRAIN_CONFIG="${TMP_CONFIG}"
    fi
fi

# Environment
export SEMANTICKITTI_DATASET="${DATASET_ROOT}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/ldm:${ROOT_DIR}/vae:${PYTHONPATH:-}"
# Help PyTorch mitigate fragmentation in the CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

echo "Using config: ${TRAIN_CONFIG}"
echo "Starting training..."
echo

if (( NUM_GPUS > 1 )); then
    # Multi-GPU: run from repo root
    cd "${ROOT_DIR}"
    accelerate launch \
        --mixed_precision="${MIXED_PRECISION}" \
        --num_processes "${NUM_GPUS}" \
        ldm/train_unconditional.py \
        --cfg "${TRAIN_CONFIG}"
else
    # Single GPU: run inside ldm for relative imports of configs/*
    cd "${ROOT_DIR}/ldm"
    python train_unconditional.py --cfg "${TRAIN_CONFIG}"
fi

echo
echo "=========================================="
echo "Training completed or interrupted"
echo "=========================================="
echo "Config used: ${TRAIN_CONFIG}"
echo "Check the configured output_dir for logs and checkpoints."
echo
