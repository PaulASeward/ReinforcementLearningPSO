import os


class Config(object):
    network_type = "DQN"
    algorithm = "PSO"
    use_mock_data = True

    # PSO PARAMETERS
    dim = 30
    swarm_size = 50
    num_episodes = 10
    num_swarm_obs_intervals = 10
    swarm_obs_interval_length = 60

    observation_length = 150
    num_actions = 5
    # action_names = ['Do nothing', 'Reset slower half', 'Encourage social learning', 'Discourage social learning', 'Reset all particles', 'Reset all particles and keep global best', 'Decrease Threshold for Replacement', 'Increase Threshold for Replacement']
    action_names = ['Do nothing', 'Decrease Threshold for Replacement', 'Increase Threshold for Replacement']

    train_steps = 20000
    log_interval = 200
    eval_interval = 500

    # EXPERIMENT PARAMETERS
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    # Output files
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Model/Checkpoint Files
    model_dir = os.path.join(results_dir, "saved_session", "network_models")
    checkpoint_dir = os.path.join(results_dir, "saved_session", "model_checkpoints")
    log_dir = os.path.join(results_dir, "saved_session", "logs")

    # EPSILON GREEDY PARAMETERS
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

    dir_save = "saved_session/"
    restore = False

    # random_start = 10
    # test_step = 5000

    # LEARNING PARAMETERS
    discount_factor = 0.01
    gamma = 0.99
    learning_rate = 0.00025
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
        self.env_action_counts = None
        self.env_action_values = None
        self.fitness_plot_path = None
        self.average_returns_plot_path = None
        self.fitness_path = None
        self.average_returns_path = None
        self.loss_file = None
        self.figure_file_right_action = None
        self.figure_file_left_action = None
        self.interval_actions_plot_path = None
        self.results_left_actions = None
        self.interval_actions_counts_path = None
        self.results_right_actions = None
        self.experiment = None
        self.num_eval_intervals = None
        self.label_iterations_intervals = None
        self.iteration_intervals = None
        self.iterations = None
        self.policy = None

    def update_properties(self, network_type=None, func_num=None, num_actions=None, num_episodes=None, num_swarm_obs_intervals=None, swarm_obs_interval_length=None, train_steps=None):
        if func_num is not None:
            self.func_num = func_num

        if num_actions is not None:
            self.num_actions = num_actions

        if num_episodes is not None:
            self.num_episodes = num_episodes
            self.trace_length = num_episodes if num_episodes < 20 else 20

        if num_swarm_obs_intervals is not None:
            self.num_swarm_obs_intervals = num_swarm_obs_intervals

        if swarm_obs_interval_length is not None:
            self.swarm_obs_interval_length = swarm_obs_interval_length

        if train_steps is not None:
            self.train_steps = train_steps
            self.log_interval = train_steps // 1000  # normal is 100
            self.eval_interval = train_steps // 400  # Normal is 40
            self.iterations = range(0, train_steps, self.eval_interval)
            self.num_eval_intervals = train_steps // self.eval_interval
            self.iteration_intervals = range(self.eval_interval, train_steps + self.eval_interval, self.eval_interval)
            self.label_iterations_intervals = range(0, train_steps + self.eval_interval, self.train_steps // 20)

        if network_type is not None:
            self.network_type = network_type
            experiment = self.network_type + "_" + self.algorithm + "_F" + str(self.func_num)
            self.experiment = experiment
            self.interval_actions_counts_path = os.path.join(self.results_dir, f"interval_actions_counts.csv")
            # self.interval_actions_plot_path = os.path.join(self.results_dir, f"interval_actions_plot.png")
            self.loss_file = os.path.join(self.results_dir, f"average_training_loss.csv")
            self.average_returns_path = os.path.join(self.results_dir, f"average_returns.csv")
            # self.average_returns_plot_path = os.path.join(self.results_dir, f"average_returns_plot.png")
            self.fitness_path = os.path.join(self.results_dir, f"average_fitness.csv")
            # self.fitness_plot_path = os.path.join(self.results_dir, f"average_fitness_plot.png")
            self.env_action_values = os.path.join(self.results_dir, f"env_actions_values.csv")
            self.env_action_counts = os.path.join(self.results_dir, f"env_actions_counts.csv")


class PSOConfig(Config):
    algorithm = "PSO"
    topology = 'global'

# def save_config(config_file, config_dict):
#     with open(config_file, 'w') as fp:
#         json.dump(config_dict, fp)
