#!/bin/bash

# Configuration
NUM_GPUS=4
DATASET="uadfV"
DATA_DIR="/path/to/UADFV"
SAVE_DIR="./checkpoints"

# Model paths (update these)
MODEL_PATHS=(
    "/path/to/model1.pt"
    "/path/to/model2.pt"
    "/path/to/model3.pt"
)

# Model names
MODEL_NAMES=(
    "ResNet50-Pruned-1"
    "ResNet50-Pruned-2"
    "ResNet50-Pruned-3"
)

# Training hyperparameters
EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
NUM_MEMBERSHIPS=3
SEED=42

# Create save directory
mkdir -p $SAVE_DIR

# Build command
CMD="torchrun --nproc_per_node=$NUM_GPUS train.py \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --model_paths ${MODEL_PATHS[@]} \
    --model_names ${MODEL_NAMES[@]} \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_memberships $NUM_MEMBERSHIPS \
    --save_dir $SAVE_DIR \
    --seed $SEED"

# Print command
echo "Running command:"
echo $CMD
echo ""

# Execute
eval $CMD
