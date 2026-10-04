#!/usr/bin/env bash
# Multi-GPU Distributed Data Parallel (DDP) training for MSQNet
set -e

DATASET=${1:-animalkingdom}
MODEL=${2:-msqnet}
DATA_DIR=${3:-./datasets}
BATCH_SIZE=${4:-8}
EPOCHS=${5:-100}
FRAMES=${6:-16}

echo "=== Distributed Training MSQNet ==="
echo "Dataset   : $DATASET"
echo "Model     : $MODEL"
echo "Data Dir  : $DATA_DIR"

python multi-label-action-main/dist_main.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --data_dir "$DATA_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --total_length "$FRAMES" \
    --distributed True \
    --train True
