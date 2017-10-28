import itertools
from collections import namedtuple

import numpy as np

from bootcamp.exceptions import StopTraining

Episode = namedtuple('Episode', ['observations', 'actions', 'rewards'])

class Trainer(object):
    def __init__(self, env, agent, callbacks=[]):
        self.env = env
        self.agent = agent
        self.callbacks = callbacks
        
    def run(self, min_batch_size, max_episode_steps=None, max_iterations=None):
        for i in itertools.count():
            if max_iterations and (i == max_iterations): break
            batch = self._simulate(min_batch_size, max_episode_steps)
            self.agent.train(batch)
            state = {'iteration': i, 'batch': batch}
            try:
                for callback in self.callbacks: callback(self, state)
            except StopTraining:
                break

    def _simulate(self, min_batch_size, max_episode_steps):
        batch = []
        timesteps = 0
        if max_episode_steps is None: max_episode_steps = self.env.spec.max_episode_steps
        while timesteps < min_batch_size:
            observations, actions, rewards = [], [], []
            observation = self.env.reset()
            for t in range(max_episode_steps):
                timesteps += 1
                observations.append(observation)
                action = self.agent.act(observation)
                actions.append(action)
                observation, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                if done: break
            episode = Episode(np.array(observations), np.array(actions), np.array(rewards))
            batch.append(episode)
        return batch
