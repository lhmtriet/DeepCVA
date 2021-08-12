#!/bin/bash
#SBATCH --partition batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --array=1-12
#SBATCH --err="_logs/generate_sequential_features_%a.err"
#SBATCH --output="_logs/generate_sequential_features_%a.out"
#SBATCH --job-name="MyJobArray"

## Setup Python Environment
module load arch/haswell
module load Anaconda3/5.0.1
module load Java
module load Singularity
module load git/2.21.0-foss-2016b
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p infer_features_sequential.csv`
python3 -u infer_features_sequential.py "${par[0]}" "${par[1]}" "${par[2]}" "${par[3]}"
