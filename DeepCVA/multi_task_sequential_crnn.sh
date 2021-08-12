#!/bin/bash
#SBATCH -p v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-1
#SBATCH --gres=gpu:1
#SBATCH --err="_logs/multitask_sequential_crnn_%a.err"
#SBATCH --output="_logs/multitask_sequential_crnn_%a.out"
#SBATCH --job-name="MyJobArray"

## Setup Python Environment
module load arch/haswell
module load Anaconda3/2020.07
module load CUDA/10.2.89
module load Java
module load Singularity
module load git/2.21.0-foss-2016b

source activate main
source deactivate main
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p multi_task_sequential_crnn.csv`
python3 -u multi_task_sequential_crnn.py "${par[0]}" "${par[1]}" "${par[2]}"