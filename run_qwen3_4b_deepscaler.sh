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

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 <seq_len> <alpha> <beta_inv> <mode> <normalization> <weight>"
  exit 1
fi

seq_len="$1"
alpha="$2"
beta_inv="$3"
mode="$4"
normalization="$5"
weight="$6"
model_name="Qwen/Qwen3-4B-Instruct-2507"

model_slug="${model_name##*/}"
model_slug="${model_slug//\//-}"

config_path="examples/configs/dist_sharpening/rloo_deepscaler_${seq_len}.yaml"
run_name="${model_slug}_deepscaler_${seq_len}_${mode}_${alpha}_${beta_inv}_${weight}_${normalization}"

log_dir="logs/dist_sharpening/${run_name}"
checkpoint_dir="/home/s/sarthmit/links/projects/aip-bengioy/sarthmit/dist_sharpening/${run_name}"

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
