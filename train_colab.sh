#!/bin/bash
# Google Colab QwenVL Training Script

# ======================
# Google Colab Environment Setup
# ======================
# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Set working directory (Colab starts in /content)
cd /content

# ======================
# Distributed Configuration for Colab
# ======================
# Colab typically has 1 GPU, but let's auto-detect
NPROC_PER_NODE=1
echo "Number of GPUs detected: $NPROC_PER_NODE"

# For single GPU training (most common in Colab)
if [ "$NPROC_PER_NODE" -eq 1 ]; then
    echo "Single GPU training detected"
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=$(shuf -i 20000-29999 -n 1)
else
    echo "Multi-GPU training detected"
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=$(shuf -i 20000-29999 -n 1)
fi

# ======================
# Path Configuration for Colab
# ======================
# All paths should be in /content or /tmp for persistence
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"    # Adjust to your model location
DATASET_NAME="checks%10"                  # Replace with your actual dataset
OUTPUT_DIR="/content/checkpoints"                # Colab persistent storage
CACHE_DIR="/tmp/cache"                           # Use /tmp for cache (faster)

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# ======================
# Colab-Optimized Training Command
# ======================
if [ "$NPROC_PER_NODE" -eq 1 ]; then
    # Single GPU - no need for torchrun
    echo "Starting single GPU training..."
    python /content/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py \
        --model_name_or_path "$MODEL_PATH" \
        --dataset_use "$DATASET_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --tune_mm_llm False \
        --tune_mm_vision True \
        --tune_mm_mlp True \
        --bf16 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 8e-6 \
        --model_max_length 1024 \
        --num_train_epochs 1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --weight_decay 0.01 \
        --logging_steps 10 \
        --save_steps 100 \
        --save_total_limit 2 \
        --dataloader_num_workers 0 \
        --remove_unused_columns False \
        --report_to "none" \
        --do_eval True \
        --evaluation_strategy epochs \
        --per_device_eval_batch_size 1 \
        --metric_for_best_model "eval_loss" \
        --load_best_model_at_end True \
        --greater_is_better False \
        
else
    # Multi-GPU with torchrun
    echo "Starting multi-GPU training..."
    torchrun --nproc_per_node=$NPROC_PER_NODE \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             /content/qwenvl/train/train_qwen.py \
             --model_name_or_path "$MODEL_PATH" \
             --dataset_use "$DATASET_NAME" \
             --output_dir "$OUTPUT_DIR" \
             --cache_dir "$CACHE_DIR" \
             --tune_mm_llm True \
             --tune_mm_vision False \
             --tune_mm_mlp False \
             --bf16 \
             --per_device_train_batch_size 2 \
             --gradient_accumulation_steps 8 \
             --learning_rate 2e-7 \
             --model_max_length 2048 \
             --num_train_epochs 1 \
             --warmup_ratio 0.03 \
             --lr_scheduler_type "cosine" \
             --weight_decay 0.01 \
             --logging_steps 10 \
             --save_steps 100 \
             --save_total_limit 2 \
             --dataloader_num_workers 2 \
             --remove_unused_columns False \
             --report_to "none"
fi

echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"