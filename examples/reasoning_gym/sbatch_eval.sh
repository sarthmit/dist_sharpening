#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2:59:00
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#
# Evaluate a reasoning_gym checkpoint on each RG category split.
#
# Mirrors the dist_sharpening training-time runtime setup (module loads +
# `uv run`) so the eval behaves identically to the training validation pass.
#
# Usage:
#   sbatch examples/reasoning_gym/sbatch_eval.sh \
#     --ckpt /path/to/step_N \
#     [--save /path/to/results] \
#     [--val-size 1000] \
#     [--vllm-tp 1]
#
# --ckpt may be either:
#   - an HF-format directory (anything that AutoModel.from_pretrained accepts), or
#   - a NeMo-RL step_N directory (sharded async-HF or DCP); we auto-convert
#     it to <ckpt>/hf/ via the appropriate converter and use that.
#
# Any flag without a matching case below is forwarded verbatim as a Hydra
# override to run_eval_rg.py (e.g. ++eval.greedy=false ++eval.temperature=0.6).

source ~/.zshrc
set -euo pipefail

module load python/3.12.4
module load cuda/12.6
module load httpproxy/1.0
module load arrow/18.1.0
module load gcc
module load opencv/4.12.0
module load rust

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN"
fi
export TOKENIZERS_PARALLELISM=false

REPO_DIR="/home/s/sarthmit/links/scratch/Projects/dist_sharpening"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
CKPT=""
SAVE=""
VAL_SIZE=""
VLLM_TP=""
CONFIG_OVERRIDE=""
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)        CKPT="$2";            shift 2 ;;
    --save)        SAVE="$2";            shift 2 ;;
    --val-size)    VAL_SIZE="$2";        shift 2 ;;
    --vllm-tp)     VLLM_TP="$2";         shift 2 ;;
    --config)      CONFIG_OVERRIDE="$2"; shift 2 ;;
    *)             EXTRA_OVERRIDES+=("$1"); shift ;;
  esac
done

if [[ -z "$CKPT" ]]; then
  echo "ERROR: --ckpt <path> is required (HF dir or step_N dir)" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Auto-materialize an HF-format dir from a training checkpoint if needed.
#
# Two on-disk training formats are supported:
#   (a) DCP (Dtensor V1/V2):
#         <CKPT>/policy/weights/.metadata          (V1)
#         <CKPT>/policy/weights/model/.metadata    (V2)
#       -> convert via examples/converters/convert_dcp_to_hf.py
#   (b) Sharded async-HF (DTensor V2 worker, "save_format=hf"):
#         <CKPT>/policy/weights/model/.hf_metadata/fqn_to_file_index_mapping.json
#         <CKPT>/policy/weights/model/shard-*-model-*.safetensors
#       -> consolidate via examples/converters/consolidate_sharded_hf.py
# In both cases the result lands in <CKPT>/hf/ and is reused on rerun.
# ---------------------------------------------------------------------------
if [[ -d "$CKPT/policy/weights" && -f "$CKPT/config.yaml" && ! -f "$CKPT/config.json" ]]; then
  HF_CKPT="$CKPT/hf"
  TOK_DIR="$CKPT/policy/tokenizer"
  if [[ -f "$HF_CKPT/config.json" ]]; then
    echo "Reusing existing HF checkpoint at: $HF_CKPT"
  elif [[ -f "$CKPT/policy/weights/model/.hf_metadata/fqn_to_file_index_mapping.json" ]]; then
    echo "Consolidating sharded HF -> $HF_CKPT"
    UV_CACHE_DIR=.cache HF_HOME=.cache uv run python "$REPO_DIR/examples/converters/consolidate_sharded_hf.py" \
      --input-dir "$CKPT/policy/weights/model" \
      --output-dir "$HF_CKPT" \
      --tokenizer-dir "$TOK_DIR"
  else
    echo "Converting DCP -> HF: $CKPT/policy/weights -> $HF_CKPT"
    UV_CACHE_DIR=.cache HF_HOME=.cache uv run python "$REPO_DIR/examples/converters/convert_dcp_to_hf.py" \
      --config "$CKPT/config.yaml" \
      --dcp-ckpt-path "$CKPT/policy/weights" \
      --hf-ckpt-path "$HF_CKPT"
  fi
  CKPT="$HF_CKPT"
fi

if [[ ! -d "$CKPT" ]]; then
  echo "ERROR: checkpoint dir not found: $CKPT" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Build Hydra override list
# ---------------------------------------------------------------------------
OVERRIDES=("++eval.checkpoint=$CKPT")
[[ -n "$SAVE" ]]      && OVERRIDES+=("++eval.save_path=$SAVE")
[[ -n "$VAL_SIZE" ]]  && OVERRIDES+=("++data.default_val_size=$VAL_SIZE")
[[ -n "$VLLM_TP" ]]   && OVERRIDES+=("++policy.generation.vllm_cfg.tensor_parallel_size=$VLLM_TP")
OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

CONFIG="${CONFIG_OVERRIDE:-$REPO_DIR/examples/configs/evals/reasoning_gym_eval.yaml}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=================================================================="
echo " RG checkpoint eval  (job ${SLURM_JOB_ID:-interactive})"
echo "=================================================================="
echo "  Repo:        $REPO_DIR"
echo "  Config:      $CONFIG"
echo "  Checkpoint:  $CKPT"
[[ -n "$SAVE" ]]     && echo "  Save path:   $SAVE"
[[ -n "$VAL_SIZE" ]] && echo "  Val size:    $VAL_SIZE per category"
echo "  Overrides:   ${OVERRIDES[*]:-<none>}"
echo "=================================================================="

UV_CACHE_DIR=.cache HF_HOME=.cache uv run python "$REPO_DIR/examples/reasoning_gym/run_eval_rg.py" \
  --config "$CONFIG" \
  "${OVERRIDES[@]}"
