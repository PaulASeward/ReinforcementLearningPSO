from agents.agent import BaseAgent
import numpy as np
import tensorflow as tf

from environment.gym_env_continuous import ContinuousPsoGymEnv
from tf_agents.environments import tf_py_environment

import gymnasium as gym
from agents.utils.experience_buffer import PPOBuffer
from agents.model_networks.ppo_model import ActorNetworkModel, CriticNetworkModel
from utils.logging_utils import ContinuousActionsResultsLogger as ResultsLogger


class PPOAgent(BaseAgent):
    def __init__(self, config):
        super(PPOAgent, self).__init__(config)
        self.results_logger = ResultsLogger(config)

        self.actor_network = ActorNetworkModel(config)
        self.critic_network = CriticNetworkModel(config)

        self.trajectory_buffer = PPOBuffer(config)

    def build_environment(self):
        if self.config.swarm_algorithm == "PMSO":
            low_limit_subswarm_action_space = [self.config.w_min, self.config.c_min, self.config.c_min]
            high_limit_subswarm_action_space = [self.config.w_max, self.config.c_max, self.config.c_max]

            self.config.lower_bound = np.array(
                [low_limit_subswarm_action_space for _ in range(self.config.num_sub_swarms)], dtype=np.float32).flatten()
            self.config.upper_bound = np.array(
                [high_limit_subswarm_action_space for _ in range(self.config.num_sub_swarms)], dtype=np.float32).flatten()
        else:
            self.config.lower_bound = np.array([self.config.w_min, self.config.c_min, self.config.c_min], dtype=np.float32)
            self.config.upper_bound = np.array([self.config.w_max, self.config.c_max, self.config.c_max], dtype=np.float32)

        if self.config.use_mock_data:
            self.raw_env = gym.make("MockContinuousPsoGymEnv-v0", config=self.config)
            self.env = self.raw_env
        else:
            self.raw_env = gym.make("ContinuousPsoGymEnv-v0", config=self.config)
            self.env = self.raw_env

        self.config.state_shape = self.env.observation_space.shape

    def get_action_and_value(self, obs):
        """
        Given an observation, returns an action sampled from the policy,
        the log-prob of that action under the current policy, and the value estimate.
        """
        action, logp = self.actor_network.sample_action(obs)
        value = self.critic_network.predict(obs)
        return action.numpy()[0], logp.numpy()[0], value[0]

    def update_policy(self, data):
        """
        Runs the PPO clipping update on the actor (policy) and the
        mean-squared-error update on the critic (value function).
        """
        obs, act, adv, ret, logp_old = data['obs'], data['act'], data['adv'], data['ret'], data['logp']
        pi_losses, v_losses, kls = [], [], []

        # Update the policy for a certain number of iterations
        for _ in range(self.config.train_policy_iterations):
            loss_pi, kl = self.actor_network.train(obs, act, logp_old, adv)
            pi_losses.append(loss_pi)
            kls.append(kl)
            if kl > 1.5 * self.config.target_kl:
                # Early stopping if KL grows too big
                break

        # Update the value function for a certain number of iterations
        for _ in range(self.config.train_value_iterations):
            loss_v = self.critic_network.train(obs, ret)
            v_losses.append(loss_v)

        return np.mean(pi_losses), np.mean(v_losses), np.mean(kls if len(kls) > 0 else [0.0])

    def replay_experience(self):
        """
        Perform the PPO update: fetch the entire on-policy batch from PPOBuffer,
        then run gradient descent for a certain number of epochs on policy and value networks.
        """
        data = self.trajectory_buffer.get()
        pi_loss, value_loss, kl = self.update_policy(data)
        return pi_loss, value_loss, kl

    def train(self):
        """
        Main training loop, which runs for config.train_steps.
        Each step, gather rollouts for num_episodes episodes,
        then update the policy/critic with PPO.
        """
        with self.writer.as_default():
            for step in range(self.config.train_steps):
                actions, rewards, swarm_observations, terminal = [], [], [], False
                obs, _ = self.env.reset()

                # Do I need to reshape obs?
                # current_state = obs
                current_state = tf.convert_to_tensor(obs.reshape(1, -1), dtype=tf.float32)

                while not terminal:
                    action, logp, value = self.get_action_and_value(current_state)
                    next_obs, reward, terminal, swarm_info = self.env.step(action)
                    self.trajectory_buffer.store(current_state, action, reward, value, logp)

                    actions.append(action)
                    rewards.append(reward)
                    swarm_observations.append(swarm_info)

                    current_state = next_obs

                # Do I need to reshape obs or current state?
                # last_val = self.critic_network.predict(tf.convert_to_tensor(current_state.reshape(1, -1), dtype=tf.float32))[0] if not terminal else 0
                last_val = self.critic_network.predict(current_state)[0] if not terminal else 0
                self.trajectory_buffer.finish_path(last_val)

                # Now update policy and value function using PPO
                pi_loss, value_loss, kl = self.replay_experience()

                self.results_logger.save_log_statements(
                    step=step + 1,
                    actions=actions,
                    rewards=rewards,
                    train_loss=[pi_loss + value_loss],
                    epsilon=0,
                    swarm_observations=swarm_observations,
                    actor_losses=pi_loss,
                    critic_losses=value_loss
                )

                if kl > 1.5 * self.config.target_kl:
                    print(f"Early stopping at step {step + 1} due to reaching max KL.")
                    break

            self.results_logger.print_execution_time()

