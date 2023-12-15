import numpy as np
import random
from collections import deque


class ExperienceBufferBase:
    def __init__(self, buffer_size, num_elements):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.num_elements = num_elements

    def add(self, experience): # Should this be a Sample object type?
        self.buffer.append(experience)

    def sample(self, *args, **kwargs):
        raise NotImplementedError('This method should be overriden.')

    def get_replay_experience(self, num_samples=None, **kwargs):

        # # REPLAY MEMORY: https://moodle31.upei.ca/pluginfile.php/1094391/mod_resource/content/0/TensorFlow%202%20Reinforcement%20Learning%20%282021%29.pdf (pg. 131)
        # if num_samples is None:
        #     num_samples = self.num_elements
        # states, actions, rewards, next_states, done = self.sample(kwargs=kwargs)
        # targets = self.target_model.predict(states)
        # next_q_values = self.target_model.predict(next_states).max(axis=1)
        # targets[range(args.batch_size), actions] = (rewards + (1 - done) * next_q_values * args.gamma)
        # self.model.train(states, targets)

        return np.reshape(np.array(self.buffer), [len(self.buffer), self.num_elements])


class ExperienceBuffer(ExperienceBufferBase):
    def __init__(self, buffer_size=200000, num_elements=5):  # Stores steps
        super().__init__(buffer_size, num_elements)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, self.num_elements])


class RecurrentExperienceBuffer(ExperienceBufferBase):
    def __init__(self, buffer_size=5000, num_elements=5):  # Stores episodes
        super().__init__(buffer_size, num_elements)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [episode for episode in self.buffer if len(episode) > trace_length-1]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) - (trace_length-1))
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, self.num_elements])


class StackedExperienceBuffer(ExperienceBufferBase):
    """
    Will use later to test lstm vs fixed size stacked frame approach
    """

    def __init__(self, buffer_size=5000, num_elements=5): # Stores episodes
        super().__init__(buffer_size, num_elements)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [episode for episode in self.buffer if len(episode) > trace_length-1]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) - (trace_length-1))
            stacked_states = []
            stacked_next_states = []
            for trace in episode[point:point + trace_length]:
                stacked_states.extend(trace[0])
                stacked_next_states.extend(trace[3])
            trace_to_return = episode[point + trace_length-1].copy()
            trace_to_return[0] = stacked_states
            trace_to_return[3] = stacked_next_states
            sampledTraces.append(trace_to_return)
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size, self.num_elements])