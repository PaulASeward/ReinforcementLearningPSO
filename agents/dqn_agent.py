from agents.agent import BaseAgent
import numpy as np
# from tqdm import tqdm

class DQNAgent(BaseAgent):
    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
