#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=15:00:00
#SBATCH --mem=24000MB
#SBATCH --array=1-44
#SBATCH --err="_logs/twb_%a.err"
#SBATCH --output="_logs/twb_%a.out"
#SBATCH --job-name="MyJobArray"

## Setup Python Environment
module load Anaconda3/5.0.1
module load Java
module load Singularity
module load git/2.21.0-foss-2016b
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
python3 get_combs.py

IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p train_feature_model_ast.csv`
python3 -u train_feature_model_ast.py "${par[0]}" "${par[1]}" "${par[2]}"
