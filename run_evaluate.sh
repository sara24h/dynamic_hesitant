#!/bin/bash

# Configuration
NUM_GPUS=4
DATASET="uadfV"
DATA_DIR="/path/to/UADFV"
CHECKPOINT="./checkpoints/best_hesitant_fuzzy.pt"

# Model paths (update these - must match training)
MODEL_PATHS=(
    "/path/to/model1.pt"
    "/path/to/model2.pt"
    "/path/to/model3.pt"
)

# Model names (must match training)
MODEL_NAMES=(
    "ResNet50-Pruned-1"
    "ResNet50-Pruned-2"
    "ResNet50-Pruned-3"
)

# Evaluation parameters
BATCH_SIZE=32
NUM_MEMBERSHIPS=3

# Build command
CMD="torchrun --nproc_per_node=$NUM_GPUS evaluate.py \
    --checkpoint $CHECKPOINT \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --model_paths ${MODEL_PATHS[@]} \
    --model_names ${MODEL_NAMES[@]} \
    --batch_size $BATCH_SIZE \
    --num_memberships $NUM_MEMBERSHIPS"

# Print command
echo "Running evaluation:"
echo $CMD
echo ""

# Execute
eval $CMD
