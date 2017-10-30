import numpy as np
import torch
from torch.autograd import Variable

from bootcamp.agents import Agent
from bootcamp.utils import compute_return, make_tensor, make_variable, normalize

class ContinuousReinforce(Agent):
    def __init__(self, approximator, stds, optimizer, baseline=None):
        self._approximator = approximator
        self._stds = stds
        self._optimizer = optimizer
        self._baseline = baseline

    def train(self, batch):
        self._optimizer.zero_grad()
        x = self._build_inputs(batch)
        y = self._build_targets(batch)
        returns = self._build_returns(batch)
        advantages = returns
        means = self._approximator(x)
        if self._baseline is not None:
            baselines = self._build_baselines(x, returns)
            advantages = returns - baselines
        log_probs = -(torch.prod(self._stds) + (((y - means) ** 2) / self._stds).sum(1)) / 2
        weights = make_variable(normalize(advantages))
        objective = -(log_probs * weights).mean()
        objective.backward()
        self._optimizer.step()

    def act(self, observation):
        x = make_variable(observation)
        means = self._approximator(x)
        return torch.normal(means, self._stds).data.numpy()

    def _build_inputs(self, batch):
        observations = np.concatenate([episode.observations for episode in batch])
        return make_variable(observations)

    def _build_targets(self, batch):
        actions = np.concatenate([episode.actions for episode in batch])
        return make_variable(actions)

    def _build_returns(self, batch):
        return np.array([compute_return(episode.rewards[t:], gamma=0.9) for episode in batch for t in range(episode.rewards.shape[0])])

    def _build_baselines(self, x, returns):
        approximator, loss, optimizer, num_iters = self._baseline
        targets = make_variable(normalize(returns))
        for i in range(num_iters):
            optimizer.zero_grad()
            baselines = approximator(x).squeeze()
            output = loss(baselines, targets)
            output.backward()
            optimizer.step()
        baselines = approximator(x).squeeze()
        return_mean, return_std = np.mean(returns), np.std(returns)
        return (return_std * baselines.data.numpy()) + return_mean
