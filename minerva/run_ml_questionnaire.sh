#!/bin/bash
#BSUB -J ml_catax
#BSUB -P acc_vascbrain
#BSUB -q gpu
#BSUB -n 6
#BSUB -R "rusage[mem=16000]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#BSUB -W 16:00
#BSUB -oo /sc/arion/work/riccig01/during/RbdQuestionnaire/minerva/output_errors/ml_questionnaire.out
#BSUB -eo /sc/arion/work/riccig01/during/RbdQuestionnaire/minerva/output_errors/ml_questionnaire.err
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
python -u /sc/arion/work/riccig01/during/CataplexyQuestionnaire/main_cv.py
