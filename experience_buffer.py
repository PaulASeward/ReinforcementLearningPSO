import numpy as np
import random
from collections import deque


class ExperienceBufferBase:
    def __init__(self, buffer_size, num_elements):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.num_elements = num_elements # [state,action,reward,next_state,done]

    def add(self, experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)


class ExperienceBufferTutorial(ExperienceBufferBase):
    def __init__(self, buffer_size=10000, num_elements=5):  # Stores steps
        super().__init__(buffer_size, num_elements)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))

        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)

        return states, actions, rewards, next_states, done


class ExperienceBufferRecurrentTutorial(ExperienceBufferBase):
    def __init__(self, buffer_size=10000, num_elements=5):  # Stores steps
        super().__init__(buffer_size, num_elements)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))

        return states, actions, rewards, next_states, done


class ExperienceBuffer(ExperienceBufferBase):
    def __init__(self, buffer_size=200000, num_elements=5):  # Stores steps
        super().__init__(buffer_size, num_elements)

    def sample(self, batch_size):
        return np.reshape(np.array(random.sample(self.buffer, batch_size)), [batch_size, self.num_elements])


class RecurrentExperienceBuffer(ExperienceBufferBase):
    def __init__(self, buffer_size=5000, num_elements=5):  # Stores episodes
        super().__init__(buffer_size, num_elements)

    def sample(self, batch_size, experience_trace_length):  # Choose random episodes with enough length, then choose random traces from those episodes.
        sampled_traces = []

        # Filter episodes with sufficient length (Here, PSO will have constant length episodes, technically unnecessary)
        valid_episodes = [episode for episode in self.buffer if len(episode) > experience_trace_length-1]
        sampled_episodes = random.sample(valid_episodes, batch_size)

        for episode in sampled_episodes:
            episode_start_index = np.random.randint(0, len(episode) - (experience_trace_length-1))
            sampled_traces.append(episode[episode_start_index: episode_start_index + experience_trace_length])

        return np.reshape(np.array(sampled_traces), [batch_size * experience_trace_length, self.num_elements])  # Do I even need to be reshaping?




# class StackedExperienceBuffer(ExperienceBufferBase):
#     """
#     Will use later to test lstm vs fixed size stacked frame approach
#     """
#
#     def __init__(self, buffer_size=5000, num_elements=5): # Stores episodes
#         super().__init__(buffer_size, num_elements)
#
#     def sample(self, batch_size, experience_trace_length):
#         tmp_buffer = [episode for episode in self.buffer if len(episode) > experience_trace_length-1]
#         sampled_episodes = random.sample(tmp_buffer, batch_size)
#         sampledTraces = []
#         for episode in sampled_episodes:
#             episode_start_index = np.random.randint(0, len(episode) - (experience_trace_length-1))
#             stacked_states = []
#             stacked_next_states = []
#             for trace in episode[episode_start_index:episode_start_index + experience_trace_length]:
#                 stacked_states.extend(trace[0])
#                 stacked_next_states.extend(trace[3])
#             trace_to_return = episode[episode_start_index + experience_trace_length-1].copy()
#             trace_to_return[0] = stacked_states
#             trace_to_return[3] = stacked_next_states
#             sampledTraces.append(trace_to_return)
#         sampledTraces = np.array(sampledTraces)
#         return np.reshape(sampledTraces, [batch_size, self.num_elements])