#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=05:59:00
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4

set -euo pipefail

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

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 <seq_len> <checkpoint_dir|-> <best|latest|base> [extra_overrides...]"
  echo "Examples:"
  echo "  $0 4096 /path/to/checkpoints latest"
  echo "  $0 4096 - base"
  exit 1
fi

seq_len="$1"
checkpoint_dir="$2"
ckpt_choice="${3:-latest}"
if [[ "$#" -ge 3 ]]; then
  shift 3
else
  shift 2
fi

case "$ckpt_choice" in
  best|latest|base) ;;
  *)
    echo "Invalid checkpoint selector: $ckpt_choice (must be 'best', 'latest', or 'base')"
    exit 1
    ;;
esac

config_path="examples/configs/dist_sharpening/rloo_deepscaler_${seq_len}.yaml"
if [[ ! -f "$config_path" ]]; then
  echo "Config not found: $config_path"
  exit 1
fi

echo "Config: ${config_path}"
echo "Checkpoint select: ${ckpt_choice}"

if [[ "$ckpt_choice" == "base" ]]; then
  UV_CACHE_DIR=.cache HF_HOME=.cache uv run python examples/run_rloo_eval.py \
    --config "$config_path" \
    --base \
    grpo.num_generations_per_prompt=16 \
    "$@"
else
  echo "Checkpoint dir: ${checkpoint_dir}"
  UV_CACHE_DIR=.cache HF_HOME=.cache uv run python examples/run_rloo_eval.py \
    --config "$config_path" \
    --checkpoint-dir "$checkpoint_dir" \
    "--${ckpt_choice}" \
    grpo.num_generations_per_prompt=16 \
    "$@"
fi
