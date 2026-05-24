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

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 <alpha> <beta_inv> <mode> <normalization> <weight>"
  exit 1
fi

alpha="$1"
beta_inv="$2"
mode="$3"
normalization="$4"
weight="$5"
seq_len=8192
model_name="Qwen/Qwen3-1.7B"

model_slug="${model_name##*/}"

# Dist-sharpening on reasoning_gym. grpo_reasoning_gym.yaml inherits
# rloo_deepscaler_8192.yaml, so loss_fn.dist_sharpening and the 8192 seq length
# are already wired; run_rl.py routes through rloo.py where DistSharpeningReward
# is applied. The reasoning_gym dataset is loaded from the precached cache_dir.
config_path="examples/reasoning_gym/grpo_reasoning_gym.yaml"
run_name="${model_slug}_reasoning_gym_${seq_len}_${mode}_${alpha}_${beta_inv}_${weight}_${normalization}"

log_dir="logs/dist_sharpening/${run_name}"
account="${SLURM_JOB_ACCOUNT:-aip-bengioy}"
checkpoint_dir="/home/s/sarthmit/links/projects/${account}/sarthmit/dist_sharpening/${run_name}"

mkdir -p "$log_dir" "$checkpoint_dir"

echo "Run name: ${run_name}"
echo "Config: ${config_path}"
echo "Log dir: ${log_dir}"
echo "Checkpoint dir: ${checkpoint_dir}"

UV_CACHE_DIR=.cache HF_HOME=.cache uv run python examples/run_rl.py \
  --config "$config_path" \
  policy.model_name="$model_name" \
  loss_fn.dist_sharpening.alpha="$alpha" \
  loss_fn.dist_sharpening.beta_inv="$beta_inv" \
  loss_fn.dist_sharpening.mode="$mode" \
  loss_fn.dist_sharpening.normalization="$normalization" \
  loss_fn.dist_sharpening.weight="$weight" \
  logger.log_dir="$log_dir" \
  logger.wandb.project="Dist-Sharpening" \
  logger.wandb.name="$run_name" \
  checkpointing.checkpoint_dir="$checkpoint_dir"
