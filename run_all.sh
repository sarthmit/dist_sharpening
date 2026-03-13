#!/bin/bash

# for runner in run_qwen_3b.sh run_llama_3b.sh; do
#   sbatch --account=aip-glaj "$runner" 1.0 0.0 optimization sum 0.0
#   # sbatch "$runner" 1.0 0.0 optimization weighted_mean 0.5
#   # sbatch "$runner" 1.0 0.0 optimization mean 0.0
#   sbatch --account=aip-glaj "$runner" 0.0 1.0 optimization sum 0.0

#   for beta_inv in 0.0 100.0 10000.0; do
#     sbatch --account=aip-glaj "$runner" 4.0 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 4.0 "$beta_inv" sampling weighted_mean 0.5

#     sbatch --account=aip-glaj "$runner" 2.0 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 2.0 "$beta_inv" sampling weighted_mean 0.5

#     sbatch --account=aip-glaj "$runner" 1.0 "$beta_inv" sampling sum 0.0

#     sbatch "$runner" 1.25 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 1.25 "$beta_inv" sampling weighted_mean 0.5

#     sbatch "$runner" 0.5 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 0.5 "$beta_inv" sampling weighted_mean 0.5
#   done

# done


# for runner in run_qwen3_4b_deepscaler.sh; do
#   sbatch --account=aip-glaj "$runner" 4096 1.0 0.0 optimization sum 0.0
#   # sbatch "$runner" 1.0 0.0 optimization weighted_mean 0.5
#   # sbatch "$runner" 1.0 0.0 optimization mean 0.0
#   # sbatch --account=aip-glaj "$runner" 0.0 1.0 optimization sum 0.0

#   for beta_inv in 0.0 10000.0; do
#     sbatch --account=aip-glaj "$runner" 4096 4.0 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 4.0 "$beta_inv" sampling weighted_mean 0.5

#     sbatch --account=aip-glaj "$runner" 4096 2.0 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 2.0 "$beta_inv" sampling weighted_mean 0.5

#     sbatch --account=aip-glaj "$runner" 4096 1.0 "$beta_inv" sampling sum 0.0

#     sbatch "$runner" 4096 1.25 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 1.25 "$beta_inv" sampling weighted_mean 0.5

#     sbatch "$runner" 4096 0.5 "$beta_inv" sampling sum 0.0
#     # sbatch "$runner" 0.5 "$beta_inv" sampling weighted_mean 0.5
#   done

# done

for runner in run_qwen_3b_fixed.sh run_llama_3b_fixed.sh; do
  sbatch --account=aip-glaj "$runner" 1.0 0.0 optimization sum 0.0
  # sbatch "$runner" 1.0 0.0 optimization weighted_mean 0.5
  # sbatch "$runner" 1.0 0.0 optimization mean 0.0
  sbatch --account=aip-glaj "$runner" 0.0 1.0 optimization sum 0.0

  for beta_inv in 0.0 100.0 10000.0; do
    sbatch --account=aip-glaj "$runner" 4.0 "$beta_inv" sampling sum 0.0
    # sbatch "$runner" 4.0 "$beta_inv" sampling weighted_mean 0.5

    sbatch --account=aip-glaj "$runner" 2.0 "$beta_inv" sampling sum 0.0
    # sbatch "$runner" 2.0 "$beta_inv" sampling weighted_mean 0.5

    sbatch --account=aip-glaj "$runner" 1.0 "$beta_inv" sampling sum 0.0

    sbatch "$runner" 1.25 "$beta_inv" sampling sum 0.0
    # sbatch "$runner" 1.25 "$beta_inv" sampling weighted_mean 0.5

    sbatch "$runner" 0.5 "$beta_inv" sampling sum 0.0
    # sbatch "$runner" 0.5 "$beta_inv" sampling weighted_mean 0.5
  done

done
