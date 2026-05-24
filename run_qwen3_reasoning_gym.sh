#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=23:59:00
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4

source ~/.zshrc
module load python/3.12.4
module load cuda/12.6
module load httpproxy/1.0
module load arrow/18.1.0
module load gcc
module load opencv/4.12.0
module load rust

hf auth login --token "$HF_TOKEN"
export TOKENIZERS_PARALLELISM=false

# Optional first arg: a run tag so concurrent runs don't clobber each other.
# Any remaining args are forwarded verbatim as Hydra overrides to run_grpo.py.
tag="${1:-uniform}"
shift || true

model_name="Qwen/Qwen3-1.7B"

model_slug="${model_name##*/}"
model_slug="${model_slug//\//-}"

config_path="examples/reasoning_gym/grpo_reasoning_gym.yaml"
run_name="${model_slug}_reasoning_gym_${tag}"

log_dir="logs/reasoning_gym/${run_name}"
checkpoint_dir="/home/s/sarthmit/links/projects/aip-bengioy/sarthmit/dist_sharpening/${run_name}"

mkdir -p "$log_dir" "$checkpoint_dir"

echo "Run name: ${run_name}"
echo "Config: ${config_path}"
echo "Log dir: ${log_dir}"
echo "Checkpoint dir: ${checkpoint_dir}"

UV_CACHE_DIR=.cache HF_HOME=.cache uv run python examples/run_grpo.py \
  --config "$config_path" \
  policy.model_name="$model_name" \
  logger.log_dir="$log_dir" \
  logger.wandb.project="Reasoning-Gym" \
  logger.wandb.name="$run_name" \
  checkpointing.checkpoint_dir="$checkpoint_dir" \
  "$@"
