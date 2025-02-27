#!/bin/bash
#SBATCH --account=def-bolufe
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --gpus-per-node=1
#SBATCH --time=159:59:59
#SBATCH --job-name=f19_ddpg
#SBATCH --output=%x-%j.out
source ~/scratch/TF_RL/bin/activate
python main.py --network_type=DDPG --swarm_algorithm=PSO --func_num=19 --num_actions=5 --num_episodes=20 --num_swarm_obs_intervals=10 --swarm_obs_interval_length=30