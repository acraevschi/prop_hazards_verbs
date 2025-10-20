#!/bin/bash
#SBATCH --qos=medium
#SBATCH --time=47:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB

echo $@

module load anaconda3
source activate hmm_irregularity

srun Rscript analysis/run_stan_model.R $@