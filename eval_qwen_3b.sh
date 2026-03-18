#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=5:59:00
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

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN"
else
  echo "HF_TOKEN is not set; using existing Hugging Face auth state if available."
fi
export TOKENIZERS_PARALLELISM=false

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 <alpha> <beta_inv> <mode> <normalization> <weight> <seq_len>"
  exit 1
fi

alpha="$1"
beta_inv="$2"
mode="$3"
normalization="$4"
weight="$5"
seq_len="$6"
model_name="Qwen/Qwen2.5-3B-Instruct"

model_slug="${model_name##*/}"
model_slug="${model_slug//\//-}"
run_name="${model_slug}_${mode}_${alpha}_${beta_inv}_${weight}_${normalization}"

log_dir="logs/dist_sharpening/${run_name}"
checkpoint_dir="/home/s/sarthmit/links/projects/aip-glaj/sarthmit/dist_sharpening/${run_name}"
metrics_dir="metrics"

mkdir -p "$log_dir" "$checkpoint_dir" "$metrics_dir"

echo "Run name: ${run_name}"
echo "Log dir: ${log_dir}"
echo "Checkpoint dir: ${checkpoint_dir}"
echo "Seq len: ${seq_len}"

eval_config="examples/configs/dist_sharpening/eval_math_lighteval.yaml"

for dataset_name in math500 minerva math_lighteval; do
  echo "Dataset: ${dataset_name}"
  for checkpoint_target in best last; do
    case "$checkpoint_target" in
      best) checkpoint_flag="--best";;
      last) checkpoint_flag="--last";;
      *) echo "Invalid checkpoint target: ${checkpoint_target}"; exit 1;;
    esac

    metrics_json_path="${metrics_dir}/${dataset_name}/${run_name}/${seq_len}/${checkpoint_target}.json"
    eval_log_dir="${log_dir}/eval_${dataset_name}_${seq_len}_${checkpoint_target}"

    echo "Evaluating: ${checkpoint_target}"
    echo "Metrics JSON: ${metrics_json_path}"
    mkdir -p "$(dirname "$metrics_json_path")"

    UV_CACHE_DIR=.cache HF_HOME=.cache uv run python examples/eval.py \
      --config "$eval_config" \
      ${checkpoint_flag} \
      --metrics-json-path "${metrics_json_path}" \
      policy.model_name="$model_name" \
      policy.max_total_sequence_length="$seq_len" \
      data.validation.dataset_name="$dataset_name" \
      logger.log_dir="$eval_log_dir" \
      checkpointing.checkpoint_dir="$checkpoint_dir"
  done
done
