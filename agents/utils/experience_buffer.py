import numpy as np
import random
from collections import deque


class ExperienceBufferBase:
    def __init__(self, config):
        pass

    def add(self, experience):
        raise NotImplementedError

    def size(self):
        # Return error if not implemented
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def batch_update(self, tree_idx, abs_errors):
        pass


class ExperienceBufferDeque(ExperienceBufferBase):
    def __init__(self, config):
        self.buffer_size = config.buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        super().__init__(config)

    def add(self, experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)


class ExperienceBufferStandard(ExperienceBufferDeque):
    def __init__(self, config):  # Stores steps
        super().__init__(config)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))

        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)

        return states, actions, rewards, next_states, done


class ExperienceBufferRecurrent(ExperienceBufferDeque):
    def __init__(self, config):  # Stores steps
        super().__init__(config)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))

        return states, actions, rewards, next_states, done



"""
Code modified from:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
"""


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, experience):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = experience  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class ExperienceBufferPriority(ExperienceBufferBase):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Experience Buffer class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """

    def __init__(self, config):
        self.capacity = config.replay_priority_capacity
        self.tree = SumTree(self.capacity)
        self.sample_count = 0
        self.epsilon = config.replay_priority_epsilon
        self.alpha = config.replay_priority_alpha
        self.beta = config.replay_priority_beta
        self.beta_increment_per_sampling = config.replay_priority_beta_increment
        self.abs_err_upper = config.replay_priority_beta_max_abs_error
        super().__init__(config)

    def size(self):
        return self.sample_count

    def add(self, experience):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, experience)  # set the max p for new p
        self.sample_count = min(self.sample_count + 1, self.tree.capacity)

    def sample(self, batch_size):
        b_idx = np.empty((batch_size,), dtype=np.int32)
        # b_memory = np.empty((batch_size, len(self.tree.data[0])))
        b_memory = np.empty((batch_size,), dtype=object)
        ISWeights = np.empty((batch_size, 1))
        pri_seg = self.tree.total_p / batch_size  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        a = self.tree.tree[-self.tree.capacity:]
        min_prob = np.min(a[a != 0]) / self.tree.total_p  # for later calculate ISweight
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            # b_idx[i], b_memory[i, :] = idx, data
            b_idx[i] = idx
            b_memory[i] = data

        return b_idx, b_memory, ISWeights.astype(np.float32)

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
