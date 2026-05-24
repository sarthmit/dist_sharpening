#!/usr/bin/env bash
# Submit one sbatch_eval.sh job per Qwen3-1.7B reasoning_gym dist_sharpening
# checkpoint step.
#
# The training script (run_qwen3_reasoning_gym_sharpening.sh) keeps the
# best-by-val:accuracy step and the latest step on disk (keep_top_k=1 +
# exclude_latest). For runs where the latest step is also the best, only
# step_250 exists — best == last, one eval is enough. For runs that
# degraded past their best, both step_150 (best) and step_250 (last) are
# present — we eval both.
#
# Results land at:
#   $RESULTS_ROOT/<run_name>/<step>/summary.json
# (mirrors PFN-RL's eval result layout).
#
# Usage:
#   examples/reasoning_gym/launch_eval_qwen3_1.7b.sh                # submits all
#   DRY_RUN=1 examples/reasoning_gym/launch_eval_qwen3_1.7b.sh      # prints, no submit

set -euo pipefail

REPO_DIR="/home/s/sarthmit/links/scratch/Projects/dist_sharpening"
SBATCH_SCRIPT="$REPO_DIR/examples/reasoning_gym/sbatch_eval.sh"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/s/sarthmit/links/projects/aip-bengioy/sarthmit/dist_sharpening}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/eval_results/reasoning_gym}"

DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_DIR"

submitted=0
skipped=0
missing=0
for run_dir in "$CHECKPOINT_ROOT"/Qwen3-1.7B_reasoning_gym_*/; do
  [[ -d "$run_dir" ]] || continue
  run_name="$(basename "$run_dir")"

  # Collect every step_N directory present (sorted by step number).
  mapfile -t step_dirs < <(ls -d "$run_dir"step_* 2>/dev/null | sort -t_ -k2 -n)
  if [[ ${#step_dirs[@]} -eq 0 ]]; then
    echo "MISSING (no step_*): $run_name"
    missing=$((missing + 1))
    continue
  fi

  for step_dir in "${step_dirs[@]}"; do
    step="$(basename "$step_dir")"
    save="$RESULTS_ROOT/$run_name/$step"
    if [[ -f "$save/summary.json" ]]; then
      echo "SKIP (summary exists): $run_name/$step"
      skipped=$((skipped + 1))
      continue
    fi
    mkdir -p "$save"
    cmd=(sbatch
      --job-name="rg-eval-${run_name}-${step}"
      "$SBATCH_SCRIPT"
      --ckpt "$step_dir"
      --save "$save")
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY: ${cmd[*]}"
    else
      "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
  done
done

echo
echo "Submitted: $submitted   Skipped: $skipped   Missing: $missing"
