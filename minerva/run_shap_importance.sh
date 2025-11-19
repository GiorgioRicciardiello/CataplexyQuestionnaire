#!/bin/bash
#BSUB -J ml_catax_shap
#BSUB -P acc_vascbrain
#BSUB -q gpu
#BSUB -n 10
#BSUB -R "rusage[mem=32000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#BSUB -W 10:00
#BSUB -oo /sc/arion/work/riccig01/during/CataplexyQuestionnaire/minerva/output_errors/ml_shap.out
#BSUB -eo /sc/arion/work/riccig01/during/CataplexyQuestionnaire/minerva/output_errors/ml_shap.err
#BSUB -L /bin/bash

# Load modules
module purge
module load anaconda3/2024.06
module load cuda/11.8

# Activate existing conda environment
source activate rbd_env

# Debug info
echo "Running on $(hostname)"
echo "CPUs: $LSB_DJOB_NUMPROC"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
python --version

# Run script
python -u /sc/arion/work/riccig01/during/CataplexyQuestionnaire/feature_importance.py
