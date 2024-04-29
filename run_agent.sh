#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --gpus-per-node=1
#SBATCH --time=159:00:00
#SBATCH --job-name=f6_drqn_random
#SBATCH --output=%x-%j.out
source ~/scratch/TF_RL2/bin/activate
python main.py --network_type=DRQN --func_num=6 --num_actions=3 --num_episodes=20 --num_swarm_obs_intervals=10 --swarm_obs_interval_length=30