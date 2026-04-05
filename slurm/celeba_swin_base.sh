#!/bin/bash
#SBATCH --job-name=dro_swin_celeba
#SBATCH -G a100:1
#SBATCH -c 18
#SBATCH --mem 96G
#SBATCH --partition=public
#SBATCH -t 1-00:00:00

module purge
module load mamba/latest
source activate dro_venv

SEEDS=(40 41 42 43 44)

cd ../

echo "==============================================="
echo "Running Swin Base GroupDRO on CelebA experiments"
echo "==============================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running seed ${SEED}"

    python run_expt.py configs/celeba_swin_base_dro.yaml --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "    Completed seed ${SEED}"
    else
        echo "    Failed seed ${SEED}"
    fi
done

echo ""
echo "Done!"
