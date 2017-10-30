import numpy as np
import torch
from torch.nn import MSELoss
from torch.autograd import Variable

from bootcamp.agents import Agent
from bootcamp.utils import compute_return, make_tensor, make_variable, normalize

class ActorCritic(Agent):
    def __init__(self, actor_approximator, stds, actor_optimizer, critic_approximator, critic_optimizer, critic_loss=None, critic_iterations=5, gamma=1.0):
        self._actor_approximator = actor_approximator
        self._stds = stds
        self._actor_optimizer = actor_optimizer
        self._critic_approximator = critic_approximator
        self._critic_optimizer = critic_optimizer
        self._critic_loss = critic_loss or MSELoss()
        self._critic_iterations = critic_iterations
        self._gamma = gamma

    def train(self, batch):
        self._actor_optimizer.zero_grad()
        x = self._build_inputs(batch)
        y = self._build_targets(batch)
        rewards = np.concatenate([episode.rewards for episode in batch])
        returns = self._build_returns(batch)
        means = self._actor_approximator(x)
        values = self._train_critic(x, returns)
        log_probs = -(torch.prod(self._stds) + (((y - means) ** 2) / self._stds).sum(1)) / 2
        advantages = rewards + self._gamma * np.concatenate([values[1:], np.zeros(1)]) - values
        weights = make_variable(normalize(advantages))
        objective = -(log_probs * weights).mean()
        objective.backward()
        self._actor_optimizer.step()

    def act(self, observation):
        x = make_variable(observation)
        means = self._actor_approximator(x)
        return torch.normal(means, self._stds).data.numpy()

    def _build_inputs(self, batch):
        observations = np.concatenate([episode.observations for episode in batch])
        return make_variable(observations)

    def _build_targets(self, batch):
        actions = np.concatenate([episode.actions for episode in batch])
        return make_variable(actions)

    def _build_returns(self, batch):
        return np.array([compute_return(episode.rewards[t:], self._gamma) for episode in batch for t in range(episode.rewards.shape[0])])

    def _train_critic(self, x, returns):
        y = make_variable(normalize(returns))
        for _ in range(self._critic_iterations):
            self._critic_optimizer.zero_grad()
            y_hat = self._critic_approximator(x).squeeze()
            output = self._critic_loss(y_hat, y)
            output.backward()
            self._critic_optimizer.step()
        y_hat = self._critic_approximator(x).squeeze()
        return_mean, return_std = np.mean(returns), np.std(returns)
        return (return_std * y_hat.data.numpy()) + return_mean
