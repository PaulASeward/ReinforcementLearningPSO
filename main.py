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
    parser.add_argument("--train", type=str, default=False, help="Whether to train a network or to examine a given network")
    parser.add_argument("--steps", type=int, default=20000, help="number of iterations to train")
    args, remaining = parser.parse_known_args()

    if args.algorithm == "PSO":
        config = PSOConfig()
        config.algorithm = args.algorithm
    else:
        print("Unsupported algorithm type: ", args.algorithm)
        sys.exit(1)

    assert args.network_type in ["DQN", "DRQN"], "Please specify a network_type of either DQN or DRQN"
    assert args.func_num in list(range(1, 29)), "Please specify a func_num from 1-28"

    config.update_properties(network_type=args.network_type, func_num=args.func_num, train_steps=args.steps)
    config.train = args.train

    print("==== Experiment: ", config.experiment)
    print("==== Args used:")
    print(args)
    print("==== Remaining (Unknown) Args:")

    print(remaining)

    main = Main(config)

    if config.train:
        print(">> Training mode. Number of Steps to Train:", config.train_steps)
        print(">> Evalutation Interval:", config.eval_interval)
        print(">> Logging Interval:", config.log_interval)
        print(">> Number of Episodes per Iteration:", config.num_eval_episodes)

        main.train()
    else:
        print("Making Plots")
        main.plot()
        # print(">> Evaluation mode. Number of Episodes to Evaluate:", config.train_steps)
        # main.evaluate(config.number_evaluations, config.checkpoint_dir)



