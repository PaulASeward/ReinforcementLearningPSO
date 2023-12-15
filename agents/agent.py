import numpy as np

class BaseAgent:
    def __init__(self, config):
        self.config = config

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.states = np.zeros([args.time_steps, self.state_dim])

        self.model = DRQN(self.state_dim, self.action_dim)
        self.target_model = DRQN(self.state_dim, self.action_dim)

        self.update_target_model_weights()
        self.buffer = ReplayBuffer()

    def update_target_model_weights(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)

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

                self.states = np.zeros([args.time_steps,  self.state_dim])
                self.update_states(self.env.reset())

                while not done:
                    action = self.model.get_action(self.states)

                    next_state, reward, done, _ = self.env.step(action)
                    prev_states = self.states

                    self.update_states(next_state)
                    self.buffer.store(prev_states, action, reward * 0.01, self.states, done)

                    episode_reward += reward

                if self.buffer.size() >= args.batch_size:
                    self.replay_experience()
                self.update_target_model_weights()
                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)