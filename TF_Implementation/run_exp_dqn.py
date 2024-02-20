# Import statements
from __future__ import absolute_import, division, print_function
from PSOEnv import PSOEnv

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


# EXPERIMENT PARAMETERS
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

func_num = 19
dim = 30
experiment = "DQN_PSO"

# Execution parameters
num_iterations = 20000
initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
log_interval = 200
num_eval_episodes = 10
eval_interval = 1000

# Creating environments
environment = PSOEnv(func_num, dimension=dim, minimum=fDeltas[func_num - 1])

# Print Array Specifications for Obs, Rewards, Actions
print('Observation Spec:')
print(environment.time_step_spec().observation)
print('Reward Spec:')
print(environment.time_step_spec().reward)
print('Action Spec:')
print(environment.action_spec())

# Practice one step:
time_step = environment.reset()
print('Time step:')
print(time_step)
action = np.array(1, dtype=np.int32)
next_time_step = environment.step(action)
print('Next time step:')
print(next_time_step)

# LOAD TWO ENVIRONMENTS
# Usually two environments are instantiated: one for training and one for evaluation
train_py_env = environment  # Python environments for training
eval_py_env = environment  # and evaluation...

train_env = tf_py_environment.TFPyEnvironment(train_py_env)  # TF environments for training
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)  # and evaluation (Random Policy requires a TF environment)

# SET UP Q-Network
# Define a helper function to create Dense layers configured with the right activation and kernel initializer.

fc_layer_params = (100, 75, 50, 25)
action_tensor_spec = tensor_spec.from_spec(environment.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
print(f"num_actions: {num_actions}")


def make_dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


dense_layers = [make_dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

# # According to the TF tutorial this old version has to be used for checkpoints
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
# global_step = tf.compat.v1.train.get_or_create_global_step()

# Creating agent
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

# POLICIES

eval_policy = agent.policy
collect_policy = agent.collect_policy

# Random Policy for collecting the initial data
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
time_step = train_env.reset()  # The time step for the random policy must be from a TF environment
random_policy.action(time_step)

# # Example Policy action call
# example_environment = tf_py_environment.TFPyEnvironment(Envi=)
# time_step = example_environment.reset()
# print(random_policy.action(time_step))

#
# def compute_avg_return(environment, policy, num_episodes=10):
#     total_return = 0.0
#     total_fitness = 0.0
#     for _ in range(num_episodes):
#
#         time_step = environment.reset()
#         episode_return = 0.0
#
#         while not time_step.is_last():
#             action_step = policy.action(time_step)
#             time_step = environment.step(action_step.action)
#             episode_return += time_step.reward
#         total_return += episode_return
#         total_fitness += environment.pyenv.envs[0]._best_fitness
#
#     avg_return = total_return / num_episodes
#     avg_fitness = total_fitness / num_episodes
#     return avg_return, avg_fitness


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# REPLAY BUFFER (TRAJECTORY) AND DATA COLLECTION
table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
    agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
print("arrived here1")
table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)
print("arrived here2")
reverb_server = reverb.Server([table])
print("arrived here3")
replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)
print("arrived here4")
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)
print("arrived here5")
print(agent.collect_data_spec)
print("arrived here6")

# Data Collection

# This driver collects initial data using the random policy
py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

# Creating the dataset pipeline to give access to the data
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# TRAINING THE AGENT

# During this step we do two things: Collect data from environment and use it to train NN
# Here we are evaluating the policy after every _(eval_interval)_ runs, and printing a sneakpeek every _(log_interval)_ run.


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

for _ in range(num_iterations):
  # Collect a few steps and save to the replay buffer.
  time_step, _ = collect_driver.run(time_step)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)


# VISUALIZATION

# Use matplotlib.pyplot to chart how the policy improved during training.

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.show()



#
# # Policy Saver
# save_dir = os.getcwd()
# policy_dir = os.path.join(save_dir, 'policy')
# tf_policy_saver = policy_saver.PolicySaver(agent.policy)
#
# # Checkpoint
# # checkpoint_directory = "/tmp/training_checkpoints"
# # checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
# checkpoint_dir = os.path.join(save_dir, 'checkpoint')
# train_checkpointer = common.Checkpointer(
#     ckpt_dir=checkpoint_dir,
#     max_to_keep=1,
#     agent=agent,
#     policy=agent.policy,
#     replay_buffer=replay_buffer,
#     global_step=global_step
# )
#
# train_checkpointer.initialize_or_restore()
# global_step = tf.compat.v1.train.get_global_step()
#
# returns = []
# fitness = []
# loss = []
# if os.path.exists(results_file_reward):
#     # load existing data
#     file = open(loss_file, "r")
#     loss.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
#     loss = [item for sublist in loss for item in sublist]
#     file.close()
#
#     file = open(results_file_reward, "r")
#     returns.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
#     returns = [item for sublist in returns for item in sublist]
#     file.close()
#
#     file = open(results_file_fitness, "r")
#     fitness.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
#     fitness = [item for sublist in fitness for item in sublist]
#     file.close()
# else:
#     # Evaluate the agent's policy once before training and create new lists
#     avg_return, avg_fitness = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
#     returns.append(float(avg_return))
#     fitness.append(avg_fitness)
#
# for _ in range(num_iterations):
#     time_step, _ = collect_driver.run(time_step)
#     experience, unused_info = next(iterator)
#     train_loss = agent.train(experience).loss
#     step = agent.train_step_counter.numpy()
#     train_checkpointer.save(global_step)
#     tf_policy_saver.save(policy_dir)
#     if step % log_interval == 0:
#         print('step = {0}: loss = {1}'.format(step, train_loss))
#         loss.append(train_loss)
#         numpy.savetxt(loss_file, loss, delimiter=", ", fmt='% s')
#
#     if step % eval_interval == 0:
#         avg_return, avg_fitness = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
#         print('step = {0}: Average Return = {1}'.format(step, avg_return))
#         returns.append(float(avg_return))
#         fitness.append(avg_fitness)
#
#         # saving the f19_drqn_10_10_60_results into a TEXT file
#         # pickle.dump(returns, open(results_file_reward, "wb"))
#         numpy.savetxt(results_file_reward, returns, delimiter=", ", fmt='% s')
#         # numpy.savetxt(results_file_fitness, fitness, delimiter=", ", fmt='% s')
#
# # saving the rewards plot into a file
# iterations = range(0, num_iterations + 1, eval_interval)
# plt.plot(iterations, returns)
# plt.ylabel('Average Return')
# plt.xlabel('Iterations')
# plt.savefig(figure_file_rewards, dpi='figure', format="png", metadata=None,
#             bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto')
# plt.close()
#
# # saving the fitness plot into a file
# iterations = range(0, num_iterations + 1, eval_interval)
# plt.plot(iterations, fitness)
# plt.ylabel('Best Fitness')
# plt.xlabel('Iterations')
# plt.savefig(figure_file_fitness, dpi='figure', format="png", metadata=None,
#             bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto')
# plt.close()
#
# print(f"--- Execution took {(time.time() - start_time) / 3600} hours ---")
