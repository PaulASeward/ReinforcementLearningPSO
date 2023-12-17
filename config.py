import os


class Config(object):
    func_num = 19
    dim = 30
    experiment = "DQN_PSO_F19"
    network_type = "DQN"

    # PSO PARAMETERS
    # observation_length = 150
    # swarm_size = 30
    action_dim = 5
    state_dim = 150
    num_iterations = 20000
    initial_collect_steps = 100
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    # batch_size = 64
    # learning_rate = 1e-3
    log_interval = 200
    num_eval_episodes = 10
    eval_interval = 500
    iterations = range(0, num_iterations + 1, eval_interval)

    # EXPERIMENT PARAMETERS
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    # Output files
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    loss_file = os.path.join(results_dir, f"{experiment}_loss.csv")
    results_file_reward = os.path.join(results_dir, f"{experiment}_returns.csv")
    results_file_fitness = os.path.join(results_dir, f"{experiment}_fitness.csv")
    figure_file_rewards = os.path.join(results_dir, f"{experiment}_plot.png")
    figure_file_fitness = os.path.join(results_dir, f"{experiment}_fit_plot.png")
    results_action_values = os.path.join(results_dir, f"{experiment}_actions_values.csv")
    results_action_counts = os.path.join(results_dir, f"{experiment}_actions_counts.csv")
    results_right_actions = os.path.join(results_dir, f"{experiment}_right_action_counts.csv")
    results_left_actions = os.path.join(results_dir, f"{experiment}_left_action_counts.csv")
    figure_file_left_action = os.path.join(results_dir, f"{experiment}_left_actions_plot.png")
    figure_file_right_action = os.path.join(results_dir, f"{experiment}_right_actions_plot.png")

    # Model/Checkpoint Files
    model_dir = os.path.join(results_dir, "saved_session", "network_models")
    checkpoint_dir = os.path.join(results_dir, "saved_session", "model_checkpoints")
    log_dir = os.path.join(results_dir, "saved_session", "logs")

    # EPSILON GREEDY PARAMETERS
    epsilon_start = 1.0
    epsilon_end = 0.02
    epsilon_decay_episodes = 1000000
    epsilon_decay = float((epsilon_start - epsilon_end)) / float(epsilon_decay_episodes)

    # DQN TRAINING PARAMETERS
    batch_size = 64
    trace_length = 10
    train_steps = 50000000
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
    number_evaluations = 10000


class PSOConfig(Config):
    algorithm = "PSO"
    topology = 'global'

# def save_config(config_file, config_dict):
#     with open(config_file, 'w') as fp:
#         json.dump(config_dict, fp)
