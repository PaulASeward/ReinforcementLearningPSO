"""Core classes."""

import numpy as np
import os
import random


class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the `action` in `state`. Expected to be the same type/dimensions as the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal


class ReplayMemory:
    """Interface for replay memories.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true terminal state (i.e. the env returned is_terminal=True), of it
      is an artificial terminal state (i.e. agent quit the episode early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will implement a different method of choosing the samples.
      Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, config):
        """Setup memory.
        Collections.deque class (Once the memory fills up oldest values should be removed.) can be used as the underlying storage, but sample method is very slow.

        Instead, use a list as a ring buffer. Just track the index where the next sample should be inserted in the list.
        """
        self.config = config

        self.actions = np.empty(self.config.mem_size, dtype=np.int32)
        self.rewards = np.empty(self.config.mem_size, dtype=np.float32)
        self.states = np.empty((self.config.mem_size, self.config.observation_length, self.config.swarm_size), dtype=np.uint8)
        self.terminals = np.empty((self.config.mem_size,), dtype=np.float16)

        self.current_count = 0  # This maxes at the mem_size
        self.current_idx = 0  # This will go up to mem_size, then reset to 0
        self.dir_save = config.dir_save + "memory/"

        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)

    def save(self):
        np.save(self.dir_save + "states.npy", self.states)
        np.save(self.dir_save + "actions.npy", self.actions)
        np.save(self.dir_save + "rewards.npy", self.rewards)
        np.save(self.dir_save + "terminals.npy", self.terminals)

    def load(self):
        self.states = np.load(self.dir_save + "states.npy")
        self.actions = np.load(self.dir_save + "actions.npy")
        self.rewards = np.load(self.dir_save + "rewards.npy")
        self.terminals = np.load(self.dir_save + "terminals.npy")

    # def clear(self):
    #     self.current_idx = 0


class DQNReplayMemory(ReplayMemory):
    def __init__(self, config):
        super(DQNReplayMemory, self).__init__(config)

        self.sample_states = np.empty((self.config.batch_size, self.config.history_len, self.config.observation_length, self.config.swarm_size), dtype=np.uint8)  # what is the difference between pre and post?
        self.sample_next_states = np.empty((self.config.batch_size, self.config.history_len, self.config.observation_length, self.config.swarm_size), dtype=np.uint8)

    def get_state(self, index):
        index = index % self.current_count
        if index >= self.config.history_len - 1:
            states = self.states[(index - (self.config.history_len - 1)):(index + 1), ...]
            return states
        else:
            indices = [(index - i) % self.current_count for i in reversed(range(self.config.history_len))]
            return self.states[indices, ...]

    def add(self, state, reward, action, terminal):
        assert state.shape == (self.config.observation_length, self.config.swarm_size)  # Only keep for initial practice tests

        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.states[self.current_idx] = state
        self.terminals[self.current_idx] = terminal

        self.current_idx += 1
        self.current_count = max(self.current_count, self.current_idx)  # This increments towards the mem_size and then stops and stays at mem_size
        self.current_idx = self.current_idx % self.config.mem_size  # Ring buffer topology (current_idx will go up to mem_size, then reset to 0)

    #     max_index = min(self.current_idx, self.config.mem_size) - 1
    #
    #     for _ in range(batch_size):
    #         sample_idx = np.random.randint(self.config.history_len - 1, max_index)

    def sample_batch(self):
        assert self.current_count > self.config.history_len  # Ensures there is enough history to sample

        sample_indices = []
        max_index = min(self.current_idx, self.config.history_len)

        for i in range(self.config.batch_size):
            while True:                    # history_len is min: 5     #current_count:  100-1
                sample_idx = random.randint(self.config.history_len, self.current_count - 1)      # sample: 30

                # Here we are choosing indexes so that the consecutive indexes chosen for a sample does not cross over where the list with ring topology is rewriting at.

                #        30   >=  Current_idx: 12            30  -   5 =   25              <    12
                if sample_idx >= self.current_idx and sample_idx - self.config.history_len < self.current_idx:
                # if sample_idx >= self.current_idx > sample_idx - self.config.history_len:

                    continue  # Cannot be higher than current_idx, AND cannot be lower than current_idx - history_len

                if self.terminals[(sample_idx - self.config.history_len): sample_idx].any():
                    continue  # Sampled state shouldn't contain episode end. Re-sample

                break
            self.sample_states[i] = self.get_state(sample_idx - 1)
            self.sample_next_states[i] = self.get_state(sample_idx)
            sample_indices.append(sample_idx)

        sample_actions = self.actions[sample_indices]
        sample_rewards = self.rewards[sample_indices]
        sample_terminals = self.terminals[sample_indices]

        return self.sample_states, sample_actions, sample_rewards, self.sample_next_states, sample_terminals


class DRQNReplayMemory(ReplayMemory):
    def __init__(self, config):
        super(DRQNReplayMemory, self).__init__(config)

        self.timesteps = np.empty(self.config.mem_size, dtype=np.int32)
        self.states = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1, self.config.observation_length, self.config.swarm_size), dtype=np.uint8)

        self.actions_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.rewards_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.terminals_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))

    def add(self, state, reward, action, terminal, t):
        assert state.shape == (self.config.observation_length, self.config.swarm_size)

        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.states[self.current_idx] = state
        self.timesteps[self.current_idx] = t
        self.terminals[self.current_idx] = float(terminal)

        self.current_count = max(self.current_count, self.current_idx + 1)
        self.current_idx = (self.current_idx + 1) % self.config.mem_size

    def sample_batch(self):
        assert self.current_count > self.config.min_history + self.config.states_to_update

        for i in range(self.config.batch_size):

            while True:
                sample_idx = random.randint(self.config.min_history, self.current_count - 1)
                if sample_idx >= self.current_idx and sample_idx - self.config.min_history < self.current_idx:
                    continue
                if sample_idx < self.config.min_history + self.config.states_to_update + 1:
                    continue
                if self.timesteps[sample_idx] < self.config.min_history + self.config.states_to_update:
                    continue
                break

            self.states[i] = self.states[sample_idx - (self.config.min_history + self.config.states_to_update + 1): sample_idx]
            self.actions_out[i] = self.actions[sample_idx - (self.config.min_history + self.config.states_to_update + 1): sample_idx]
            self.rewards_out[i] = self.rewards[sample_idx - (self.config.min_history + self.config.states_to_update + 1): sample_idx]
            self.terminals_out[i] = self.terminals[sample_idx - (self.config.min_history + self.config.states_to_update + 1): sample_idx]

        return self.states, self.actions_out, self.rewards_out, self.terminals_out


    # def append(self, state, action, reward, is_terminal):
    #     self.actions[self.current_idx_count % self.memory_size] = action
    #     self.rewards[self.current_idx_count % self.memory_size] = reward
    #     self.states[self.current_idx_count % self.memory_size] = state
    #     self.terminals[self.current_idx_count % self.memory_size] = is_terminal
    #     self.current_idx_count += 1
    #
    # def get_state(self, index):
    #     state = self.states[index+1 - self.history_length : index+1, :, :]
    #     # history dimension last
    #     return np.transpose(state, (1, 2, 0))  # Is Transpose the right thing to do here?
    #
    # def sample(self, batch_size):
    #     samples = []
    #     # ensure enough frames to sample
    #     # assert self.current_idx_count > self.history_length
    #     max_index = min(self.current_idx, self.config.mem_size) - 1
    #
    #     for _ in range(batch_size):
    #         sample_idx = np.random.randint(self.config.history_len - 1, max_index)
    #
    #         # sampled state shouldn't contain episode end
    #         while self.terminals[sample_idx+1 - self.config.history_len: sample_idx+1].any():
    #             sample_idx = np.random.randint(self.config.history_len - 1, max_index)
    #
    #         new_sample = Sample(
    #             state=self.get_state(sample_idx),
    #             action=self.actions[sample_idx],
    #             reward=self.rewards[sample_idx],
    #             next_state=self.get_state(sample_idx + 1),
    #             is_terminal=self.terminals[sample_idx]
    #         )
    #         samples.append(new_sample)
    #     return samples
    #
    # def clear(self):
    #     self.current_idx_count = 0
