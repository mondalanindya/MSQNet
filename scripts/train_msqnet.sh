#!/usr/bin/env bash
# Train MSQNet on Animal Kingdom dataset
set -e

DATASET=${1:-animalkingdom}
MODEL=${2:-msqnet}
DATA_DIR=${3:-./datasets}
BATCH_SIZE=${4:-16}
EPOCHS=${5:-100}
FRAMES=${6:-16}

echo "=== Training MSQNet ==="
echo "Dataset   : $DATASET"
echo "Model     : $MODEL"
echo "Data Dir  : $DATA_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs    : $EPOCHS"
echo "Frames    : $FRAMES"

python run.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --data_dir "$DATA_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --total_length "$FRAMES" \
    --train True
