#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --gpus-per-node=1
#SBATCH --time=199:00:00
#SBATCH --job-name=PSO_RL_Test_FE_Distributions
#SBATCH --output=%x-%j.out
source ~/scratch/TF_RL2/bin/activate
python main.py