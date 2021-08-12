#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=1:00:00
#SBATCH --mem=8000MB
#SBATCH --array=1-132
#SBATCH --err="_logs/INFER_%a.err"
#SBATCH --output="_logs/INFER_%a.out"
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
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p infer_features_ast.csv`
python3 -u infer_features_ast.py "${par[0]}" "${par[1]}" "${par[2]}" "${par[3]}"
