#!/bin/bash

# Specify hyperparameters
runs=5

# Check if sbatch is available
if command -v sbatch > /dev/null; then
  sbatch_cmd="sbatch"
else
  echo "\033[1;31mYou are not in a slurm environment. Executing experiments sequentially!\033[0m"
  sbatch_cmd=""
fi

for seed in $(seq 0 $((runs-1))); do
    if [ -z "$sbatch_cmd" ]; then
        echo "\033[1;32mExecuting job with seed:\033[0m -s $seed"
        ./scripts/job.sh -s $seed -t layer -e mut-only -a 1
    else
        $sbatch_cmd --job-name="run-$seed" scripts/job.sh -s $seed -t layer -e mut-only -a 1
    fi
done