# Implementing the Deep Recurrent Q-Learning algorithm and DRQN agent 121
import tensorflow as tf
from datetime import datetime
import os
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
import gym
import argparse
import numpy as np
from collections import deque
import random

try:
    parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch3-DRQN")
    parser.add_argument("--env", default="CartPole-v0")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--time_steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--eps_decay", type=float,default=0.995)
    parser.add_argument("--eps_min", type=float,default=0.01)
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()
except Exception as e:
    print(f"An error occurred: {e}")
print("Parsed arguments:", args)

logdir = os.path.join(args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S"))
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size=args.batch_size, experience_trace_length=4):
        samples = random.sample(self.buffer, batch_size)  # sample.len=64

        states, actions, rewards, next_states, done = map(np.asarray, zip(*samples))

        # Below code resizes to (64,16). Likely unneeded since we want to keep the dimension of the sample as this is expected input to the model.
        # states = np.array(states).reshape(args.batch_size, -1)
        # next_states = np.array(next_states).reshape(args.batch_size, -1)

        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class DRQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        self.optimizer = Adam(args.lr)
        self.compute_loss = tf.keras.losses.MeanSquaredError()
        self.model = self.nn_model()

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((args.time_steps, self.state_dim)),
                LSTM(32, activation="tanh"),
                Dense(16, activation="relu"),
                Dense(self.action_dim),
            ]
        )

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, args.time_steps, self.state_dim])

        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)

        q_value = self.predict(state)[0]

        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        return np.argmax(q_value)

    def train(self, states, targets):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)

            assert targets.shape == logits.shape

            loss = self.compute_loss(targets, logits)
            grads = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


class Agent:
    def __init__(self, env):
        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.states = np.zeros([args.time_steps, self.state_dim])

        self.model = DRQN(self.state_dim, self.action_dim)
        self.target_model = DRQN(self.state_dim, self.action_dim)

        self.update_target()
        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)  # This is likely unnecessary as it ges rewritten

            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = (rewards + (1 - done) * next_q_values * args.gamma)

            self.model.train(states, targets)

    def update_states(self, next_state):
        self.states = np.roll(self.states, -1, axis=0)
        self.states[-1] = next_state

    def train(self, max_episodes=1000):
        with writer.as_default():
            for ep in range(max_episodes):
                done, episode_reward = False, 0

                self.states = np.zeros([args.time_steps,  self.state_dim])  # Starts with choosing an action from empty states. Uses rolling window size 4
                self.update_states(self.env.reset())

                while not done:
                    action = self.model.get_action(self.states)

                    next_state, reward, done, _ = self.env.step(action)
                    prev_states = self.states

                    self.update_states(next_state)  # Updates the states array removing oldest when adding newest for sliding window
                    self.buffer.store(prev_states, action, reward * 0.01, self.states, done)

                    episode_reward += reward

                if self.buffer.size() >= args.batch_size:
                    self.replay_experience()  # Only replay experience once there is enough in buffer to sample.

                self.update_target()  # target model gets updated AFTER episode, not during like the regular model.

                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(max_episodes=20000)


# TO RUN:
# python tutorial_example.py
# python tutorial_example.py –env "MountainCar-v0"