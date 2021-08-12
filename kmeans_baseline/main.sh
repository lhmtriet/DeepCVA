#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem=16000MB
#SBATCH --array=1-51
#SBATCH --err="_logs/cvss_baselines_%a.err"
#SBATCH --output="_logs/cvss_baselines_%a.out"
#SBATCH --job-name="cvss_baselines"

## Setup Python Environment
module load Anaconda3/2020.07

# Strange workaround - not sure why it works. Reference:
# https://stackoverflow.com/questions/36733179/
# Without it, the wrong python version is used (default instead of conda)
source activate main
source deactivate main
source activate main

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p main.csv`
python -u main.py "${par[0]}" "${par[1]}"