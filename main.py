from agents.dqn_agent import DQNAgent
from agents.drqn_agent import DRQNAgent
from agents.ddpg_agent import DDPGAgent
from config import PSOConfig
import argparse


class Main:
    def __init__(self, config):
        agent_mapping = {
            "DQN": DQNAgent,
            "DRQN": DRQNAgent,
            "DDPG": DDPGAgent
        }

        self.agent = agent_mapping.get(config.network_type, DRQNAgent)(config)

    def train(self):
        self.agent.get_actions()
        self.agent.train()
        self.plot()

    def plot(self):
        self.agent.results_logger.plot_results()

    # def evaluate(self, num_episodes, checkpoint_dir):
    #     self.agent.evaluate(num_episodes, checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN Agent on PSO Algorithm")
    parser.add_argument("--network_type", type=str, default="DDPG", help="Type of the network to build, can either be 'DQN', 'DDPG',  or 'DRQN'")
    parser.add_argument("--swarm_algorithm", type=str, default="PMSO", help="The metaheuristic swarm algorithm to use. Currently only PSO or PMSO is supported")
    parser.add_argument("--func_num", type=int, default=11, help="The function number to optimize. Good functions to evaluate are 6,10,11,14,19")
    parser.add_argument("--dim", type=int, default=30,help="The number of dimensions in the search space. Default is 30.")
    parser.add_argument("--swarm_size", type=int, default=50, help="The number of particles in the swarm. Default is 50.")
    parser.add_argument("--num_actions", type=int, default=15, help="The number of actions to choose from in the action space. Default is 5.")
    parser.add_argument("--action_dimensions", type=int, default=15, help="The number of actions to choose from in the action space. Default is 15.")
    parser.add_argument("--num_episodes", type=int, default=20, help="The number of episodes in each Reinforcement Learning Iterations before terminating.")
    parser.add_argument("--num_swarm_obs_intervals", type=int, default=10, help="The number of swarm observation intervals. Ex) At 10 evenly spaced observation intervals, observations in the swarm will be collected. Default is 10.")
    parser.add_argument("--swarm_obs_interval_length", type=int, default=30, help="The number of observations per episode conducted in the swarm. Ex) Particle Best Replacement Counts are averaged over the last _ observations before an episode terminates and action is decided. Default is 30.")
    parser.add_argument("--train", type=bool, default=True, help="Whether to train a network or to examine a given network")
    parser.add_argument("--mock", type=bool, default=False, help="To use a mock data environment for testing")
    parser.add_argument("--priority_replay", type=bool, default=False, help="To use a priority replay buffer for training")
    parser.add_argument("--steps", type=int, default=2000, help="number of iterations to train")
    args, remaining = parser.parse_known_args()

    config = PSOConfig()
    config.train = args.train
    config.use_mock_data = args.mock
    config.use_priority_replay = args.priority_replay

    assert args.swarm_algorithm in ["PSO", "PMSO"], "Please specify a swarm_algorithm of either PSO or PMSO"
    assert args.network_type in ["DQN", "DRQN", "DDPG"], "Please specify a network_type of either DQN, DRQN, or DDPG"
    assert args.func_num in list(range(1, 29)), "Please specify a func_num from 1-28"
    assert args.dim in [2, 5, 10, 20, 20, 30, 40, 50, 60, 70, 80, 90, 100], "Please specify a dim from 2,5,10,20,30,40,50,60,70,80,90,100"

    config.update_properties(network_type=args.network_type, swarm_algorithm=args.swarm_algorithm, func_num=args.func_num, num_actions=args.num_actions,
                             action_dimensions=args.action_dimensions, swarm_size=args.swarm_size, dimensions=args.dim, num_episodes=args.num_episodes,
                             num_swarm_obs_intervals=args.num_swarm_obs_intervals, swarm_obs_interval_length=args.swarm_obs_interval_length,
                             train_steps=args.steps)

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
        max_func_eval = config.swarm_size * config.num_episodes * config.obs_per_episode
        print(f"=== Function Evaluation Budget: {config.dim} dimensions x 10 000/dim = {func_eval_budget} Function Evaluations")
        print(f"=== Observations Per Episode: {config.num_swarm_obs_intervals} Number of Swarm Observation Intervals per Episode x {config.swarm_obs_interval_length} Number of Observations in each Interval  = {config.obs_per_episode}  Observations per Episode")
        print(f"=== Function Evaluation Allocation: {config.swarm_size} Swarm Size (# Particles) x {config.num_episodes} Episodes x {config.obs_per_episode} Observations per Episode  = {max_func_eval} Function Evaluations")
        print()

        if func_eval_budget != max_func_eval:
            raise ValueError("Maximum Function Evaluation budget does not match total allocations of function evaluations.")

        main.train()
    else:
        print("Making Plots")
        main.plot()
        # print(">> Evaluation mode. Number of Episodes to Evaluate:", config.train_steps)
        # main.evaluate(config.number_evaluations, config.checkpoint_dir)
