"""Core classes."""

import numpy as np


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


# def replay_experience(self):
#     for _ in range(10):
#         states, actions, rewards, next_states, done= self.buffer.sample()
#
#         targets = self.target_model.predict(states)
#         next_q_values = self.target_model.predict(next_states).max(axis=1)
#         targets[range(args.batch_size), actions] = (rewards + (1 - done) * next_q_values * args.gamma)
#         self.model.train(states, targets)


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
    def __init__(self, args):
        """Setup memory.

        You should specify the maximum size o the memory. Once the memory fills up oldest values should be removed.
        You can try the collections.deque class as the underlying storage, but your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the index where the next sample should be inserted in the list.
        """
        self.memory_size = args.replay_memory_size
        self.history_length = args.num_frames
        self.actions = np.zeros(self.memory_size, dtype=np.int8)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.states = np.zeros((self.memory_size, args.observation_length, args.swarm_size), dtype=np.float32)
        self.terminals = np.zeros(self.memory_size, dtype=bool)
        self.current_idx_count = 0

    def append(self, state, action, reward, is_terminal):
        self.actions[self.current_idx_count % self.memory_size] = action
        self.rewards[self.current_idx_count % self.memory_size] = reward
        self.states[self.current_idx_count % self.memory_size] = state
        self.terminals[self.current_idx_count % self.memory_size] = is_terminal
        self.current_idx_count += 1

    def get_state(self, index):
        state = self.states[index+1 - self.history_length : index+1, :, :]
        # history dimension last
        return np.transpose(state, (1, 2, 0))  # Is Transpose the right thing to do here?

    def sample(self, batch_size):
        samples = []
        # ensure enough frames to sample
        # assert self.current_idx_count > self.history_length
        max_index = min(self.current_idx_count, self.memory_size) - 1

        for _ in range(batch_size):
            index = np.random.randint(self.history_length - 1, max_index)

            # sampled state shouldn't contain episode end
            while self.terminals[index+1 - self.history_length: index+1].any():
                index = np.random.randint(self.history_length - 1, max_index)

            new_sample = Sample(
                state=self.get_state(index),
                action=self.actions[index],
                reward=self.rewards[index],
                next_state=self.get_state(index + 1),
                is_terminal=self.terminals[index]
            )
            samples.append(new_sample)
        return samples

    def clear(self):
        self.current_idx_count = 0
