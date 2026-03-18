#!/bin/bash

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <seq_len>"
  exit 1
fi

seq_len="$1"

for runner in eval_qwen_3b.sh eval_llama_3b.sh eval_qwen_3b_fixed.sh eval_llama_3b_fixed.sh; do
  sbatch --account=aip-glaj "$runner" 1.0 0.0 optimization sum 0.0 "$seq_len"
  sbatch --account=aip-glaj "$runner" 0.0 1.0 optimization sum 0.0 "$seq_len"

  for beta_inv in 0.0 100.0 10000.0; do
    sbatch --account=aip-glaj "$runner" 4.0 "$beta_inv" sampling sum 0.0 "$seq_len"

    sbatch --account=aip-glaj "$runner" 2.0 "$beta_inv" sampling sum 0.0 "$seq_len"

    sbatch --account=aip-glaj "$runner" 1.0 "$beta_inv" sampling sum 0.0 "$seq_len"

    sbatch "$runner" 1.25 "$beta_inv" sampling sum 0.0 "$seq_len"

    # sbatch "$runner" 0.5 "$beta_inv" sampling sum 0.0 "$seq_len"
  done
done
