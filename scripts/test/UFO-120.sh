#!/bin/bash

# Testing script for UFO-120 dataset
# Usage: ./scripts/test/UFO-120.sh

# Testing parameters
BATCH_SIZE=4
EPOCHS=1
FINETUNE=""
DEVICE="cuda"
SEED=42
RESUME="./checkpoints/UFO-120/best.pth"
START_EPOCH=0
EVAL_FLAG="--eval"
TEST_ON_LAST_EPOCH="False"
NUM_WORKERS=4
CFG_PATH="./configs/ufo-120.yaml"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run evaluation
python main_new.py \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --finetune "$FINETUNE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume "$RESUME" \
    --start_epoch "$START_EPOCH" \
    $EVAL_FLAG \
    --test_on_last_epoch "$TEST_ON_LAST_EPOCH" \
    --num_workers "$NUM_WORKERS" \
    --cfg_path "$CFG_PATH" \
    2>&1 | tee logs/ufo120_evaluation_$(date +%Y%m%d_%H%M%S).log 