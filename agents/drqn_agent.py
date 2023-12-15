from agents.agent import BaseAgent
import numpy as np
# from tqdm import tqdm

class DRQNAgent(BaseAgent):
    def __init__(self, config):
        super(DRQNAgent, self).__init__(config)
