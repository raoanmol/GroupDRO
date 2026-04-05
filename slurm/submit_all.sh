#!/bin/bash
# Submit all SLURM job scripts in this directory
# Usage: cd slurm/ && bash submit_all.sh

DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$DIR"/*.sh; do
    [ "$(basename "$script")" = "submit_all.sh" ] && continue
    echo "Submitting $(basename "$script")..."
    sbatch "$script"
done
