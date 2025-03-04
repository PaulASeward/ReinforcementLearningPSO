import os
import copy

import numpy as np


class Config(object):
    use_discrete_env = None
    use_mock_data = False
    use_priority_replay = False
    reward_function = "smoothed_total_difference_reward"
    penalty_for_negative_reward = 0
    use_attention_layer = False
    use_ou_noise = False

    # AGENT PARAMETERS
    num_episodes = 20
    num_swarm_obs_intervals = 10
    swarm_obs_interval_length = 30
    observation_length = 151

    train_steps = 20000
    log_interval = 200
    eval_interval = 500
    test_episodes = 10

    replay_experience_length = 1

    # EXPERIMENT PARAMETERS
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    best_f_standard_pso = [0,0,0,0,0,-26,-112,-21,-26,0,-85,-120,-200,-2279,-4080,-1,-104,-130,-5,-12,-305,-2513,-4594,-276,0,0,0,0]
    swarm_improvement_pso = [0,0,0,0,2,31,20,1,0,18,2,10,20,118,1300,1,43,141,4,1,1,237,1075,1,0,0,0,0]
    standard_deviations = [0.00e+00, 7.82e+05, 7.06e+07, 4.55e+03, 0.00e+00, 4.34e+00, 1.74e+01, 5.51e-2, 1.95e+00, 5.53e-2, 1.51e+01, 1.72e+01, 2.21e+01, 3.80e+02, 6.25e+02, 3.51e-1, 1.55e+01, 2.68e+01, 1.30e+00, 5.12e-1, 5.30e+01, 5.15e+02, 7.06e+02, 5.64e+00, 7.12e+00, 4.61e+01, 7.43e+01, 2.82e-13]

    # Output files
    results_dir = "results"
    # results_dir = "run_history/20241126/f11_PMSO_DDPG"

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

    # Replay Buffer
    # buffer_size = 10000
    buffer_size = 20000
    # buffer_size = 1000000
    batch_size = 64
    replay_priority_capacity = 100000
    replay_priority_epsilon = 0.01  # small amount to avoid zero priority
    replay_priority_alpha = 0.7  # [0~1] convert the importance of TD error to priority
    replay_priority_beta = 0.5  # importance-sampling, from initial value increasing to 1
    replay_priority_beta_increment = 0.001
    replay_priority_beta_max_abs_error = 1.0  # clipped abs error

    # DRQN TRAINING PARAMETERS
    trace_length = 10

    # DDPG TRAINING PARAMETERS
    ou_mu = None  # Will be set to zeros of action_dim in update_properties
    ou_theta = 0.15
    # ou_sigma = 0.1
    ou_sigma = 0.15
    ou_dt = 1e-2

    tau = 0.001
    # tau = 0.125
    upper_bound = None
    lower_bound = None
    # actor_layers = (400, 300)
    # actor_layers = (64,32)
    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3
    # actor_learning_rate = 5e-6
    # critic_learning_rate = 5e-6
    # critic_layers = (16, 32, 48)
    actor_layers = (64, 128, 256)
    critic_layers = (64, 128, 256)
    # actor_layers = (64, 64)
    # critic_layers = (64, 64, 1)
    # critic_layers = (600, 300)
    action_dim = None
    state_shape = None
    action_bound = None
    action_shift = None

    # PPO TRAINING PARAMETERS
    clip_ratio = 0.2
    target_kl = 0.01
    lam = 0.97

    # policy_learning_rate = 3e-4
    # value_function_learning_rate = 1e-3
    # train_policy_iterations = 80
    # train_value_iterations = 80
    train_policy_iterations = 10
    train_value_iterations = 10

    actions_descriptions = None
    continuous_action_offset = None

    dir_save = "saved_session/"
    restore = False

    # LEARNING PARAMETERS
    # discount_factor = 0.01
    gamma = 0.99
    learning_rate = 0.001
    lr_method = "adam"

    # learning_rate = 0.00025
    # learning_rate_minimum = 0.00025
    # lr_decay = 0.97
    # keep_prob = 0.8

    # LSTM PARAMETERS
    num_lstm_layers = 1
    lstm_size = 512
    # min_history = 4

    # EVALUATION PARAMETERS
    # number_evaluations = 10000

    # PSO Config:
    topology = 'global'
    is_sub_swarm = False

    w = 0.729844  # Inertia weight
    # w_min = 0.33  # Min of 5 decreases of 10%
    w_min = 0.43  # Min of 5 decreases of 10%
    w_max = 1.175  # Max of 5 increases of 10%
    c1 = 2.05 * w  # Social component Learning Factor
    c2 = 2.05 * w  # Cognitive component Learning Factor
    c_min = 0.883  # Min of 5 decreases of 10%
    # c_min = 0.583  # Min of 5 decreases of 10%
    c_max = 2.409  # Max of 5 increases of 10%
    rangeF = 100
    v_min = 59.049
    v_max = 161.051
    replacement_threshold = 1.0
    replacement_threshold_min = 0.5
    replacement_threshold_max = 1.0
    replacement_threshold_decay = 0.95

    def __init__(self):
        self.func_num = None
        self.action_dimensions = None
        self.experiment_config_path = None
        self.action_counts_path = None
        self.continuous_action_history_path = None
        self.action_values_path = None
        self.test_step_results_path = None
        self.action_training_values_path = None
        self.epsilon_values_path = None
        self.fitness_plot_path = None
        self.average_returns_plot_path = None
        self.fitness_path = None
        self.episode_results_path = None
        self.training_step_results_path = None
        self.average_returns_path = None
        self.loss_file = None
        self.actor_loss_file = None
        self.critic_loss_file = None
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
        self.swarm_algorithm = None
        self.num_sub_swarms = None
        self.network_type = None
        self.dim = None

    def clone(self):
        return copy.deepcopy(self)

    def update_properties(self, network_type=None, swarm_algorithm=None, func_num=None, num_actions=None, action_dimensions=None, swarm_size=None, dimensions=None, num_episodes=None, num_swarm_obs_intervals=None, swarm_obs_interval_length=None, train_steps=None):
        if func_num is not None:
            self.func_num = func_num

        if num_actions is not None:
            self.num_actions = num_actions

        if action_dimensions is not None:
            self.action_dimensions = action_dimensions
            self.ou_mu = np.zeros(self.action_dimensions)

        if swarm_size is not None:
            self.swarm_size = swarm_size
            self.observation_length = self.swarm_size * 3 + 1

        if dimensions is not None:
            self.dim = dimensions

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

        if swarm_algorithm is not None:
            self.swarm_algorithm = swarm_algorithm
            if swarm_algorithm == "PMSO":
                self.num_sub_swarms = 5

        if network_type is not None:
            if network_type in ["DQN", "DRQN"]:
                self.use_discrete_env = True
            else:
                self.use_discrete_env = False

            self.network_type = network_type
            experiment = self.network_type + "_" + self.swarm_algorithm + "_F" + str(self.func_num)
            self.experiment = experiment
            self.experiment_config_path = os.path.join(self.results_dir, f"experiment_config.json")
            self.interval_actions_counts_path = os.path.join(self.results_dir, f"interval_actions_counts.csv")
            self.loss_file = os.path.join(self.results_dir, f"average_training_loss.csv")
            self.actor_loss_file = os.path.join(self.results_dir, f"actor_loss.csv")
            self.critic_loss_file = os.path.join(self.results_dir, f"critic_loss.csv")
            self.average_returns_path = os.path.join(self.results_dir, f"average_returns.csv")
            self.fitness_path = os.path.join(self.results_dir, f"average_fitness.csv")
            self.episode_results_path = os.path.join(self.results_dir, f"episode_results.csv")
            self.training_step_results_path = os.path.join(self.results_dir, f"step_results.csv")
            self.test_step_results_path = os.path.join(self.results_dir, f"test_results.csv")
            self.action_values_path = os.path.join(self.results_dir, f"actions_values.csv")
            self.action_training_values_path = os.path.join(self.results_dir, f"actions_training_values.csv")
            self.continuous_action_history_path = os.path.join(self.results_dir, f"continuous_action_history.npy")
            self.action_counts_path = os.path.join(self.results_dir, f"actions_counts.csv")
            self.epsilon_values_path = os.path.join(self.results_dir, f"epsilon_values.csv")
            self.standard_pso_path = os.path.join(self.standard_pso_results_dir, f"f{self.func_num}.csv")