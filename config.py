import os


class Config(object):
    network_type = "DQN"
    algorithm = "PSO"
    use_mock_data = False

    # AGENT PARAMETERS
    num_episodes = 20
    num_swarm_obs_intervals = 10
    swarm_obs_interval_length = 30
    observation_length = 150

    train_steps = 20000
    log_interval = 200
    eval_interval = 500

    # EXPERIMENT PARAMETERS
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    # Output files
    results_dir = "results"
    standard_pso_results_dir = "pso/standard_pso_results"
    os.makedirs(results_dir, exist_ok=True)
    swarm_locations_dir = os.path.join(results_dir, "swarm_locations")
    os.makedirs(swarm_locations_dir, exist_ok=True)
    env_swarm_locations_name = "swarm_locations.npy"
    env_swarm_velocities_name = "swarm_velocities.npy"
    env_swarm_best_locations_name = "swarm_best_locations.npy"
    env_swarm_evaluations_name = "swarm_evaluations.npy"
    env_meta_data_name = "meta_data.csv"

    # Model/Checkpoint Files
    model_dir = os.path.join(results_dir, "saved_session", "network_models")
    checkpoint_dir = os.path.join(results_dir, "saved_session", "model_checkpoints")
    log_dir = os.path.join(results_dir, "saved_session", "logs")

    # EPSILON GREEDY PARAMETERS
    policy = "ExponentialDecayGreedyEpsilon"
    epsilon_start = 1.0
    epsilon_end = 0.01
    # epsilon_decay_episodes = 1000
    # epsilon_decay = float((epsilon_start - epsilon_end)) / float(epsilon_decay_episodes)
    epsilon_decay = 0.995

    # DQN TRAINING PARAMETERS
    batch_size = 64
    trace_length = 10
    history_len = 4
    # frame_skip = 4
    # max_steps = 10000
    # train_freq = 8
    # update_freq = 10000
    # train_start = 20000

    # DDPG TRAINING PARAMETERS
    # ou_mean = 1
    ou_mean = 0.0
    ou_theta = 0.15
    # ou_sigma = 0.2
    ou_sigma = 0.5
    ou_dt = 1e-2
    tau = 0.005
    upper_bound = 1.0
    lower_bound = -1.0

    dir_save = "saved_session/"
    restore = False

    # random_start = 10
    # test_step = 5000

    # LEARNING PARAMETERS
    discount_factor = 0.01
    gamma = 0.99
    learning_rate = 0.001
    # learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    lr_method = "adam"
    lr_decay = 0.97
    keep_prob = 0.8

    state = None
    mem_size = 800000

    # LSTM PARAMETERS
    num_lstm_layers = 1
    lstm_size = 512
    min_history = 4
    states_to_update = 4

    # EVALUATION PARAMETERS
    # number_evaluations = 10000

    def __init__(self):
        self.func_num = None
        self.action_counts_path = None
        self.action_values_path = None
        self.fitness_plot_path = None
        self.average_returns_plot_path = None
        self.fitness_path = None
        self.average_returns_path = None
        self.loss_file = None
        self.interval_actions_counts_path = None
        self.standard_pso_path = None
        self.experiment = None
        self.num_eval_intervals = None
        self.label_iterations_intervals = None
        self.iteration_intervals = None
        self.obs_per_episode = None
        self.iterations = None
        self.swarm_size = None
        self.num_actions = None

    def update_properties(self, network_type=None, func_num=None, num_actions=None, swarm_size=None, num_episodes=None, num_swarm_obs_intervals=None, swarm_obs_interval_length=None, train_steps=None):
        if func_num is not None:
            self.func_num = func_num

        if num_actions is not None:
            self.num_actions = num_actions

        if swarm_size is not None:
            self.swarm_size = swarm_size
            self.observation_length = self.swarm_size * 3

        if num_episodes is not None:
            self.num_episodes = num_episodes
            self.trace_length = num_episodes if num_episodes < 20 else 20

        if num_swarm_obs_intervals is not None:
            self.num_swarm_obs_intervals = num_swarm_obs_intervals

        if swarm_obs_interval_length is not None:
            self.swarm_obs_interval_length = swarm_obs_interval_length

        self.obs_per_episode = self.swarm_obs_interval_length * self.num_swarm_obs_intervals

        if train_steps is not None:
            self.train_steps = train_steps
            self.log_interval = train_steps // 100  # normal is 100
            self.eval_interval = train_steps // 40  # Normal is 40
            self.iterations = range(0, train_steps, self.eval_interval)
            self.num_eval_intervals = train_steps // self.eval_interval
            self.iteration_intervals = range(self.eval_interval, train_steps + self.eval_interval, self.eval_interval)
            self.label_iterations_intervals = range(0, train_steps + self.eval_interval, self.train_steps // 20)

        if network_type is not None:
            self.network_type = network_type
            experiment = self.network_type + "_" + self.algorithm + "_F" + str(self.func_num)
            self.experiment = experiment
            self.interval_actions_counts_path = os.path.join(self.results_dir, f"interval_actions_counts.csv")
            self.loss_file = os.path.join(self.results_dir, f"average_training_loss.csv")
            self.average_returns_path = os.path.join(self.results_dir, f"average_returns.csv")
            self.fitness_path = os.path.join(self.results_dir, f"average_fitness.csv")
            self.action_values_path = os.path.join(self.results_dir, f"actions_values.csv")
            self.action_counts_path = os.path.join(self.results_dir, f"actions_counts.csv")
            self.standard_pso_path = os.path.join(self.standard_pso_results_dir, f"f{self.func_num}.csv")


class PSOConfig(Config):
    algorithm = "PSO"
    topology = 'global'

    dim = 30

    w = 0.729844  # Inertia weight
    w_min = 0.43  # Min of 5 decreases of 10%
    w_max = 1.175  # Max of 5 increases of 10%
    c1 = 2.05 * w  # Social component Learning Factor
    c2 = 2.05 * w  # Cognitive component Learning Factor
    c_min = 0.88  # Min of 5 decreases of 10%
    c_max = 2.41  # Max of 5 increases of 10%
    rangeF = 100
    v_min = 59.049
    v_max = 161.051
    replacement_threshold = 1.0
    replacement_threshold_min = 0.5
    replacement_threshold_max = 1.0
    replacement_threshold_decay = 0.95

# def save_config(config_file, config_dict):
#     with open(config_file, 'w') as fp:
#         json.dump(config_dict, fp)
