#!/bin/bash

# Training script for RangeLDM on SemanticKITTI dataset
# Make sure to update paths and settings according to your setup

# ============================================================================
# IMPORTANT: Update these paths before running!
# ============================================================================

# Path to SemanticKITTI dataset root folder
# When running in Docker, this will be mounted to /datasets/SemanticKITTI
# When running locally, use the full host path
if [ -d "/datasets/SemanticKITTI" ]; then
    export SEMANTICKITTI_DATASET="/datasets/SemanticKITTI"
else
    export SEMANTICKITTI_DATASET="/home/kx9279/Repos/container-3d-diffusion-lidar-super-resolution/lidiff/Datasets/SemanticKITTI"
fi

# Path to your trained VAE checkpoint (REQUIRED!)
# You need to train a VAE first using the VAE training pipeline
# Or update the config file directly
VAE_CHECKPOINT="path_to_your_vae_checkpoint.ckpt"

# ============================================================================
# Training Configuration
# ============================================================================

# Configuration file
CONFIG="ldm/configs/semantickitti.yaml"

# Number of GPUs to use
NUM_GPUS=1

# Mixed precision training: "no", "fp16", or "bf16"
MIXED_PRECISION="no"

# ============================================================================
# Verify paths
# ============================================================================

if [ ! -d "$SEMANTICKITTI_DATASET" ]; then
    echo "ERROR: SemanticKITTI dataset not found at: $SEMANTICKITTI_DATASET"
    echo "Please update the SEMANTICKITTI_DATASET path in this script."
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found at: $CONFIG"
    exit 1
fi

echo "=========================================="
echo "RangeLDM Training on SemanticKITTI"
echo "=========================================="
echo "Dataset path: $SEMANTICKITTI_DATASET"
echo "Config file: $CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "Mixed precision: $MIXED_PRECISION"
echo "=========================================="
echo ""

# Check if sequences exist
echo "Checking dataset structure..."
echo "Dataset path: $SEMANTICKITTI_DATASET"

# Detect the sequences path
if [ -d "$SEMANTICKITTI_DATASET/dataset/sequences" ]; then
    SEQUENCES_PATH="$SEMANTICKITTI_DATASET/dataset/sequences"
    echo "Found sequences at: $SEQUENCES_PATH"
elif [ -d "$SEMANTICKITTI_DATASET/sequences" ]; then
    SEQUENCES_PATH="$SEMANTICKITTI_DATASET/sequences"
    echo "Found sequences at: $SEQUENCES_PATH"
elif [ -d "$SEMANTICKITTI_DATASET/00" ]; then
    SEQUENCES_PATH="$SEMANTICKITTI_DATASET"
    echo "Found sequences at: $SEQUENCES_PATH"
else
    echo "ERROR: Cannot find sequences directory in $SEMANTICKITTI_DATASET"
    exit 1
fi

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
    seq_path="$SEQUENCES_PATH/$seq"
    if [ -d "$seq_path" ]; then
        velodyne_path="$seq_path/velodyne"
        if [ -d "$velodyne_path" ]; then
            num_files=$(ls -1 "$velodyne_path"/*.bin 2>/dev/null | wc -l)
            echo "  Sequence $seq: $num_files scans found"
        else
            echo "  Sequence $seq: WARNING - velodyne folder not found"
        fi
    else
        echo "  Sequence $seq: NOT FOUND"
    fi
done
echo ""

# ============================================================================
# Update VAE checkpoint in config if provided as argument
# ============================================================================

if [ "$#" -ge 1 ]; then
    VAE_CHECKPOINT=$1
    echo "Using VAE checkpoint: $VAE_CHECKPOINT"
fi

if [ ! -f "$VAE_CHECKPOINT" ] && [ "$VAE_CHECKPOINT" != "path_to_your_vae_checkpoint.ckpt" ]; then
    echo "WARNING: VAE checkpoint not found at: $VAE_CHECKPOINT"
    echo "Please train a VAE first or provide the correct path."
    echo ""
fi

# ============================================================================
# Training command
# ============================================================================

echo "Starting training..."
echo ""

# Change to the RangeLDM directory (important for imports)
cd /workspace/RangeLDM

# Set Python path to include ldm directory
export PYTHONPATH="/workspace/RangeLDM:/workspace/RangeLDM/ldm:/workspace/RangeLDM/vae:$PYTHONPATH"

if [ $NUM_GPUS -gt 1 ]; then
    # Multi-GPU training with accelerate
    accelerate launch \
        --mixed_precision=$MIXED_PRECISION \
        --num_processes=$NUM_GPUS \
        ldm/train_unconditional.py \
        --cfg $CONFIG
else
    # Single GPU training
    cd ldm && python train_unconditional.py \
        --cfg configs/semantickitti.yaml
fi

# ============================================================================
# Training completion
# ============================================================================

echo ""
echo "=========================================="
echo "Training completed or interrupted"
echo "=========================================="
echo "Check the output directory specified in $CONFIG"
echo "for checkpoints and training logs."
echo ""
