from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.td3 import td3_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.environments import suite_gym
import tf_agents

from absl import logging

root_dir = "DDPG_pendulum/"
env_name='Pendulum-v0'
_DEFAULT_REWARD_SCALE = 1.
eval_env_name=None
env_load_fn=suite_mujoco.load,
num_iterations=70000
actor_fc_layers=(128, 128)
critic_obs_fc_layers=None
critic_action_fc_layers=None
critic_joint_fc_layers=(128, 128)
num_parallel_environments=1
# Params for collect
initial_collect_steps=100
collect_steps_per_iteration=1
replay_buffer_capacity=50000
# Params for target update
target_update_tau=0.005
target_update_period=1
# Params for train
train_steps_per_iteration=1
batch_size=64
actor_learning_rate=3e-4
critic_learning_rate=3e-4
alpha_learning_rate=3e-4
td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error
gamma=0.99
reward_scale_factor=_DEFAULT_REWARD_SCALE
gradient_clipping=None
use_tf_functions=True
# Params for eval
num_eval_episodes = 10
eval_interval=1000
# Params for summaries and logging
train_checkpoint_interval=10000
policy_checkpoint_interval=5000
rb_checkpoint_interval=50000
log_interval=1000
summary_interval=1000
summaries_flush_secs=10
debug_summaries=False
summarize_grads_and_vars=False
eval_metrics_callback=None

logging.set_verbosity(logging.INFO)



def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
  del init_action_stddev
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      #std_transform=tf.nn.softplus,
      scale_distribution=True)

root_dir = os.path.expanduser(root_dir)

summary_writer = tf.compat.v2.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
summary_writer.set_as_default()

eval_metrics = [
  tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
  tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
]

global_step = tf.compat.v1.train.get_or_create_global_step()

py_env = suite_gym.load(env_name)

tf_env = tf_py_environment.TFPyEnvironment(py_env)
# create evaluation environment
eval_env_name = eval_env_name or env_name
eval_py_env = suite_gym.load(eval_env_name)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)





time_step_spec = tf_env.time_step_spec()
observation_spec = time_step_spec.observation
action_spec = tf_env.action_spec()

actor_net = tf_agents.agents.ddpg.actor_network.ActorNetwork(
    observation_spec, action_spec, fc_layer_params=actor_fc_layers,

)

critic_net = tf_agents.agents.ddpg.critic_network.CriticNetwork(
    (observation_spec, action_spec), joint_fc_layer_params=critic_joint_fc_layers)

"""actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=actor_fc_layers,
    continuous_projection_net=normal_projection_net)

critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=critic_obs_fc_layers,
    action_fc_layer_params=critic_action_fc_layers,
    joint_fc_layer_params=critic_joint_fc_layers)
"""

tf_agent = tf_agents.agents.DdpgAgent(
    time_step_spec,
    action_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=critic_learning_rate),
    # alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
    #   learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=td_errors_loss_fn,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars,
    train_step_counter=global_step)
tf_agent.initialize()


# Make the replay buffer.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=num_parallel_environments,
    max_length=replay_buffer_capacity)
replay_observer = [replay_buffer.add_batch]

env_steps = tf_metrics.EnvironmentSteps(prefix='Train')

average_return = tf_metrics.AverageReturnMetric(
    prefix='Train',
    buffer_size=num_eval_episodes,
    batch_size=tf_env.batch_size)

train_metrics = [
    tf_metrics.NumberOfEpisodes(prefix='Train'),
    env_steps,
    average_return,
    tf_metrics.AverageEpisodeLengthMetric(
        prefix='Train',
        buffer_size=num_eval_episodes,
        batch_size=tf_env.batch_size),
]

eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)

initial_collect_policy = random_tf_policy.RandomTFPolicy(
    tf_env.time_step_spec(), tf_env.action_spec())

collect_policy = tf_agent.collect_policy

train_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(root_dir, 'train'),
    agent=tf_agent,
    global_step=global_step,
    metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
policy_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(root_dir, 'policy'),
    policy=eval_policy,
    global_step=global_step)
rb_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
    max_to_keep=1,
    replay_buffer=replay_buffer)

train_checkpointer.initialize_or_restore()

rb_checkpointer.initialize_or_restore()

initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=replay_observer + train_metrics,
    num_steps=initial_collect_steps)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    collect_policy,
    observers=replay_observer + train_metrics,
    num_steps=collect_steps_per_iteration)

initial_collect_driver.run = common.function(initial_collect_driver.run)
collect_driver.run = common.function(collect_driver.run)
tf_agent.train = common.function(tf_agent.train)



# Collect initial replay data.
if env_steps.result() == 0 or replay_buffer.num_frames() == 0:
    logging.info(
      'Initializing replay buffer by collecting experience for %d steps'
      'with a random policy.', initial_collect_steps)
    initial_collect_driver.run()

