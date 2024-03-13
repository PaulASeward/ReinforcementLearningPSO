from agents.dqn_agent import DQNAgent
from agents.drqn_agent import DRQNAgent
from config import PSOConfig
import sys
import argparse


class Main:
    def __init__(self, config):
        if config.network_type == "DQN":
            self.agent = DQNAgent(config)
        else:
            self.agent = DRQNAgent(config)

    def train(self):
        self.agent.get_actions()
        self.agent.train()

    def plot(self):
        self.agent.build_plots()

    # def evaluate(self, num_episodes, checkpoint_dir):
    #     self.agent.play(num_episodes, checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN Agent on PSO Algorithm")
    parser.add_argument("--network_type", type=str, default="DRQN", help="Type of the network to build, can either be 'DQN' or 'DRQN'")
    parser.add_argument("--algorithm", type=str, default="PSO", help="The metaheuristic algorithm to use. Currently only pso is supported")
    parser.add_argument("--func_num", type=int, default=19, help="The function number to optimize. Good functions to evaluate are 6,10,11,14,19")
    parser.add_argument("--num_actions", type=int, default=6, help="The number of actions to choose from in the action space. Default is 5.")
    parser.add_argument("--num_episodes", type=int, default=20, help="The number of episodes in each Reinforcement Learning Iterations before terminating.")
    parser.add_argument("--num_swarm_obs_intervals", type=int, default=10, help="The number of swarm observation intervals. Ex) At 10 evenly spaced observation intervals, observations in the swarm will be collected.")
    parser.add_argument("--swarm_obs_interval_length", type=int, default=30, help="The number of observations per episode conducted in the swarm. Ex) Particle Best Replacement Counts are averaged over the last 60 observations before an episode terminates and action is decided.")
    parser.add_argument("--policy", type=str, default="ExponentialDecayGreedyEpsilon", help="The policy to use for action selection. Default is ExponentialDecayGreedyEpsilon")
    parser.add_argument("--train", type=str, default=True, help="Whether to train a network or to examine a given network")
    parser.add_argument("--steps", type=int, default=20000, help="number of iterations to train")
    args, remaining = parser.parse_known_args()

    if args.algorithm == "PSO":
        config = PSOConfig()
        config.algorithm = args.algorithm
        config.policy = args.policy
        config.num_actions = args.num_actions
    else:
        print("Unsupported algorithm type: ", args.algorithm)
        sys.exit(1)

    assert args.network_type in ["DQN", "DRQN"], "Please specify a network_type of either DQN or DRQN"
    assert args.func_num in list(range(1, 29)), "Please specify a func_num from 1-28"

    config.update_properties(network_type=args.network_type, func_num=args.func_num, num_actions=args.num_actions, num_episodes=args.num_episodes, num_swarm_obs_intervals=args.num_swarm_obs_intervals, swarm_obs_interval_length=args.swarm_obs_interval_length, train_steps=args.steps)
    config.train = args.train

    print("==== Experiment: ", config.experiment)
    print("==== Args used:")
    print(args)
    print("==== Remaining (Unknown) Args:")
    print(remaining)

    main = Main(config)

    if config.train:
        print(">> Training mode. Number of Steps to Train:", config.train_steps)
        print(">> Evaluation Interval:", config.eval_interval)
        print(">> Logging Interval:", config.log_interval)
        print(">> Number of Episodes per Iteration:", config.num_episodes)

        func_eval_budget = config.dim * 10000
        obs_per_episode = config.num_swarm_obs_intervals * config.swarm_obs_interval_length
        max_func_eval = config.swarm_size * config.num_episodes * obs_per_episode
        print(f"=== Function Evaluation Budget: {config.dim} dimensions x 10 000/dim = {func_eval_budget} Function Evaluations")
        print(f"=== Observations Per Episode: {config.num_swarm_obs_intervals} Number of Swarm Observation Intervals per Episode x {config.swarm_obs_interval_length} Number of Observations in each Interval  = {obs_per_episode}  Observations per Episode")
        print(f"=== Function Evaluation Allocation: {config.swarm_size} Swarm Size (# Particles) x {config.num_episodes} Episodes x {obs_per_episode} Observations per Episode  = {max_func_eval} Function Evaluations")
        print()

        if func_eval_budget != max_func_eval:
            raise ValueError("Maximum Function Evaluation budget does not match total allocations of function evaluations.")

        main.train()
    else:
        print("Making Plots")
        main.plot()
        # print(">> Evaluation mode. Number of Episodes to Evaluate:", config.train_steps)
        # main.evaluate(config.number_evaluations, config.checkpoint_dir)



