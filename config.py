import os


class Config(object):
    network_type = "DQN"
    algorithm = "PSO"

    # PSO PARAMETERS
    dim = 30
    observation_length = 150
    swarm_size = 30
    action_dim = 5
    state_dim = 150
    train_steps = 2000  # make this dynamically updated
    initial_collect_steps = 100
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    # batch_size = 64
    # learning_rate = 1e-3

    log_interval = 200
    num_eval_episodes = 10
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
    epsilon_end = 0.02
    epsilon_decay_episodes = 1000
    epsilon_decay = float((epsilon_start - epsilon_end)) / float(epsilon_decay_episodes)

    # DQN TRAINING PARAMETERS
    batch_size = 64
    trace_length = 10
    history_len = 4
    frame_skip = 4
    max_steps = 10000
    train_freq = 8
    update_freq = 10000
    train_start = 20000

    dir_save = "saved_session/"
    restore = False

    random_start = 10
    test_step = 5000

    # LEARNING PARAMETERS
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

    def update_properties(self, network_type=None, func_num=None, train_steps=None):
        if func_num is not None:
            self.func_num = func_num

        if train_steps is not None:
            self.train_steps = train_steps
            self.iterations = range(0, train_steps, self.eval_interval)
            self.iteration_intervals = range(self.eval_interval, train_steps + self.eval_interval, self.eval_interval)
            self.label_iterations_intervals = range(0, train_steps + self.eval_interval, self.eval_interval * 2)
            self.num_eval_intervals = train_steps // self.eval_interval

        if network_type is not None:
            self.network_type = network_type
            experiment = self.network_type + "_" + self.algorithm + "_F" + str(self.func_num)
            self.experiment = experiment
            self.results_right_actions = os.path.join(self.results_dir, f"{experiment}_right_action_counts.csv")
            self.results_left_actions = os.path.join(self.results_dir, f"{experiment}_left_action_counts.csv")
            self.results_actions = os.path.join(self.results_dir, f"{experiment}_action_counts.csv")
            self.figure_file_action = os.path.join(self.results_dir, f"{experiment}_actions_plot.png")
            self.figure_file_left_action = os.path.join(self.results_dir, f"{experiment}_left_actions_plot.png")
            self.figure_file_right_action = os.path.join(self.results_dir, f"{experiment}_right_actions_plot.png")
            self.loss_file = os.path.join(self.results_dir, f"{experiment}_loss.csv")
            self.results_file_reward = os.path.join(self.results_dir, f"{experiment}_returns.csv")
            self.results_file_fitness = os.path.join(self.results_dir, f"{experiment}_fitness.csv")
            self.figure_file_rewards = os.path.join(self.results_dir, f"{experiment}_plot.png")
            self.figure_file_fitness = os.path.join(self.results_dir, f"{experiment}_fit_plot.png")
            self.results_action_values = os.path.join(self.results_dir, f"{experiment}_actions_values.csv")
            self.results_action_counts = os.path.join(self.results_dir, f"{experiment}_actions_counts.csv")
            # self.results_action_counts = os.path.join(self.results_dir, f"PSO_DQN_actions_counts(f{func_num}).csv")



class PSOConfig(Config):
    algorithm = "PSO"
    topology = 'global'

# def save_config(config_file, config_dict):
#     with open(config_file, 'w') as fp:
#         json.dump(config_dict, fp)
