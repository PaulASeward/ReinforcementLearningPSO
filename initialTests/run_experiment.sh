#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=99:00:00
#SBATCH --job-name=PSOtest
#SBATCH --output=%x-%j.out
#SBATCH --gpus-per-node=1
~/pseward/scratch/TF_RL/bin
python pso_run.py
