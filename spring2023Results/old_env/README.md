# Reinforcement Learning Project with Vectorized Particle Swarm Optimization (PSO)

## Overview

This project aims to implement a Reinforcement Learning (RL) algorithm using the Particle Swarm Optimization (PSO) technique with a vectorized approach. The goal is to optimize the policy of an agent in a given environment using PSO, which is a population-based optimization algorithm inspired by the social behavior of birds flocking or fish schooling. The vectorized approach allows for efficient updates, taking advantage of parallel computing capabilities.

## Project Structure

The project is structured as follows:

1. `PSOEnv.py`: This contains the environment implementations required for RL training and evaluation.

2. `run_exp.py/`: This folder contains the RL agent implementation, including PSO with vectorized updates.

3. `functions.py &  VectorizedGlobalLocal/`: This folder contains utility functions and PSO vectorized implementation.

4. `run_agent.sh`: The shell script to run the RL training and evaluation process.

5. 'TF_Implementation/': This folder has the rough implementations of implementing TensorFlow's libraries for a vectorized approach
6.  
## Environment

The environment used for this project is the search space parametirzed in the constructor of the swarm. The state space consists of the agent's current position, and the action space includes resetting the hole swarm, the slower half of the swarm relative to the magnitude of their velocity, and allowing for standard optimization procedure.The environment also provides a fitness function (CEC_functions) to evaluate the agent's performance.

## Agent

The RL agent is implemented using PSO with a vectorized approach. In PSO, the population of particles (agents) explores the search space by updating their positions and velocities based on their own experience and the experience of their neighbors.

The vectorized approach allows us to efficiently update multiple particles' positions and velocities simultaneously, taking advantage of parallel computing capabilities. We are using a DQN agent for learning.

## Training

To train the RL agent, follow these steps:

1. Set up the environment by initializing the grid world and placing the agent and goal in appropriate positions.

2. Initialize the population of particles for PSO with random positions and velocities.

3. Perform PSO iterations, where particles update their positions and velocities based on the PSO formula.

4. Evaluate the performance of each particle using the environment's reward function.

5. Update the global best position based on the particle with the highest reward.

6. Repeat steps 3 to 5 for a specified number of iterations or until convergence.

## Evaluation

After training the RL agent, we evaluate its performance in the environment. We measure the agent's success rate in reaching the goal from different starting positions and report the average reward achieved.

## Dependencies

Dependencies

    - Python 3.7 or higher
    - TensorFlow
    - NumPy
    - Reverb (for the replay buffer)
    - Matplotlib (for plotting)

You can install the required dependencies by running pip install -r requirements.txt.

## Getting Started

To run the project, follow these steps:

1. Clone the repository: git clone https://github.com/PaulASeward/RL_PSO_Project.git

2. Install the required dependencies: pip install -r requirements.txt

3. Navigate to the project directory: cd RL_PSO_Project

4. Run the training script: python run_exp.py

### Installation
 You can install all dependencies by issuing following command:
 ```
 pip install -r requirements.txt
 ```
 This will install Tensorflow without GPU support. However, I highly recommend using Tensorflow with GPU support, otherwise training will take a very long time. For more information on this topic please see https://www.tensorflow.org/install/.
### Running
You can start training by:
```
python main.py --network_type=DQN --func_num=19 --steps=10000000
```
This will train a DQN on PSO with global topology for 10 mio observations. For more information on command line parameters please see:
```
python main.py -h
```

[//]: # (Visualizing the training process can be done using tensorboard by:)

[//]: # (```)

[//]: # (tensorboard --logdir=out)

[//]: # (```)
[//]: # (### Pretrained models)

[//]: # (A pretrained model for DQN PSO is available in `pretrained_models`)


## Conclusion

This project demonstrates the implementation of a Reinforcement Learning algorithm using Particle Swarm Optimization with a vectorized approach. The RL agent optimizes its policy in a PSO environment to achieve the best fitness value using PSO with additional actions to help further optimize. The vectorized PSO implementation ensures efficient updates and faster convergence.

Happy Learning!

---
