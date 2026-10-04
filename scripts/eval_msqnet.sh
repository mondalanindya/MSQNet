#!/usr/bin/env bash
# Evaluate MSQNet with a trained checkpoint
set -e

CHECKPOINT=${1:-"./checkpoints/msqnet_msqnet_animalkingdom.pth"}
DATASET=${2:-animalkingdom}
DATA_DIR=${3:-./datasets}
MODEL=${4:-msqnet}
FRAMES=${5:-16}

echo "=== Evaluating MSQNet ==="
echo "Checkpoint: $CHECKPOINT"
echo "Dataset   : $DATASET"
echo "Model     : $MODEL"
echo "Frames    : $FRAMES"

python run.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --total_length "$FRAMES" \
    --train False
