#!/bin/bash

# HuggingFace Model Evaluation Script for KG-R1 (Turn 5)
# Evaluates models directly from HuggingFace Hub (no local checkpoint needed)
#
# Usage:
#   ./eval_qwen_3b_cwq_f1_turn5.sh                            # Use default HF model
#   ./eval_qwen_3b_cwq_f1_turn5.sh JinyeopSong/KG-R1_test     # Specify HF model
#   ./eval_qwen_3b_cwq_f1_turn5.sh your-org/your-model cwq    # Specify model and dataset
#   ./eval_qwen_3b_cwq_f1_turn5.sh "" cwq reward_model.reward_kwargs.use_legacy_entity_f1=true

# ==================== CONFIGURATION ====================

# Default HuggingFace model to evaluate
DEFAULT_HF_MODEL="JinyeopSong/KG-R1_test"

# Default dataset to evaluate on
DEFAULT_DATASET="cwq"  # Options: "cwq" or "webqsp"

# Number of samples to evaluate (0 for all samples)
EVAL_SAMPLES=0

# K values for Pass@K evaluation
K_VALUES="1 2 3 4"

# WandB project name
WAND_PROJECT='KG-R1-Evaluation-HF'

# Validation batch size (adjust based on model size)
VAL_BATCH_SIZE=128

# Max turns for KG search
MAX_TURNS=5

# ==================== END CONFIGURATION ====================

export CUDA_VISIBLE_DEVICES=4
export DATA_DIR='data_kg'
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Initialize environment (optional - only if init script exists)
# if [ -f "/nobackup/users/yeopjin/init_general.sh" ]; then
#     source /nobackup/users/yeopjin/init_general.sh
# fi

# Change to project root
cd "$PROJECT_ROOT"

rm -rf ~/.cache/torch/triton/

# Parse command line arguments
HF_MODEL="${1:-$DEFAULT_HF_MODEL}"
DATASET_NAME="${2:-$DEFAULT_DATASET}"
EXTRA_OVERRIDES=("${@:3}")

# Calculate MAX_K for experiment naming
MAX_K=$(echo $K_VALUES | tr ' ' '\n' | sort -nr | head -1)
EXPERIMENT_NAME="${DATASET_NAME}-turn${MAX_TURNS}-eval-k${MAX_K}-n${EVAL_SAMPLES}-hf-$(date +%m%d_%H%M)"

# Display configuration
echo "==================== HF MODEL EVALUATION CONFIGURATION ===================="
echo "HuggingFace Model: $HF_MODEL"
echo "Dataset: $DATASET_NAME"
echo "Eval samples: $EVAL_SAMPLES"
echo "K values: $K_VALUES"
echo "Max turns: $MAX_TURNS"
echo "Validation batch size: $VAL_BATCH_SIZE"
echo "Experiment name: $EXPERIMENT_NAME"
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
    echo "Extra Hydra overrides: ${EXTRA_OVERRIDES[*]}"
fi
echo "=========================================================================="

# Set dataset files (relative to project root)
if [ "$DATASET_NAME" = "cwq" ]; then
    TEST_FILE="$PROJECT_ROOT/$DATA_DIR/cwq_search_augmented_initial_entities/test.parquet"
elif [ "$DATASET_NAME" = "webqsp" ]; then
    TEST_FILE="$PROJECT_ROOT/$DATA_DIR/webqsp_search_augmented_initial_entities/test.parquet"
fi

# Check if data file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    echo ""
    echo "Please set up data_kg directory first:"
    echo "  Run: python initialize.py"
    echo ""
    exit 1
fi

# Create output directory
mkdir -p "eval_results/eval_kg-r1/$EXPERIMENT_NAME"

echo ""
echo "Starting evaluation..."
echo "This will:"
echo "  1. Download model from HuggingFace (if not cached)"
echo "  2. Generate $MAX_K responses per question"
echo "  3. Compute Pass@K metrics for K=$K_VALUES"
echo "  4. Save results to eval_results/eval_kg-r1/$EXPERIMENT_NAME/"
echo ""

PYTHONUNBUFFERED=1 python -m verl.trainer.main_eval \
    n_rollout_eval=$MAX_K \
    k_values="[$(echo $K_VALUES | sed 's/ /,/g')]" \
    eval_samples=$EVAL_SAMPLES \
    mode=kg-search \
    +save_detailed_results=true \
    actor_rollout_ref.rollout.search.max_turns=$MAX_TURNS \
    reward_model.reward_kwargs.max_turns=$MAX_TURNS \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.project_name=$WAND_PROJECT \
    trainer.default_local_dir="eval_results/eval_kg-r1/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=1 \
    ray_init.num_cpus=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    data.train_files="$TEST_FILE" \
    data.val_files="$TEST_FILE" \
    data.val_batch_size=$VAL_BATCH_SIZE \
    actor_rollout_ref.model.path="$HF_MODEL" \
    +actor_rollout_ref.model.is_huggingface=true \
    trainer.resume_mode=none \
    reward_model.reward_kwargs.debug_log_dir="${EXPERIMENT_NAME}_debug.log" \
    "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee "eval_results/eval_kg-r1/${EXPERIMENT_NAME}/evaluation.log"

echo ""
echo "=========================================================================="
echo "Evaluation complete!"
echo ""
echo "Results saved to:"
echo "  - Summary: eval_results/eval_kg-r1/$EXPERIMENT_NAME/${DATASET_NAME}_passatk_results.json"
echo "  - Detailed: eval_results/eval_kg-r1/$EXPERIMENT_NAME/${DATASET_NAME}_detailed_results.jsonl"
echo "  - Log: eval_results/eval_kg-r1/$EXPERIMENT_NAME/evaluation.log"
echo "=========================================================================="
