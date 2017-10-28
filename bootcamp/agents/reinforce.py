import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax
from torch.autograd import Variable

from bootcamp.agents import Agent
from bootcamp.utils import compute_return, make_tensor, make_variable

class Reinforce(Agent):
    def __init__(self, approximator, optimizer):
        self._approximator = approximator
        self._optimizer = optimizer

    def train(self, batch):
        self._optimizer.zero_grad()
        x = self._build_inputs(batch)
        y = self._build_targets(batch)
        returns = self._build_returns(batch)
        scores = self._approximator(x)
        log_probs = torch.squeeze(torch.gather(log_softmax(scores), 1, y.view(-1, 1)))
        objective = -(log_probs * returns).mean()
        objective.backward()
        self._optimizer.step()
    
    def act(self, observation):
        x = make_variable(observation)
        probs = softmax(self._approximator(x))
        return torch.multinomial(probs, 1).data[0]

    def _build_inputs(self, batch):
        observations = np.concatenate([episode.observations for episode in batch])
        return make_variable(observations)

    def _build_targets(self, batch):
        actions = np.concatenate([episode.actions for episode in batch])
        return make_variable(actions, dtype='int64')

    def _build_returns(self, batch):
        raw_returns = np.concatenate([np.repeat(compute_return(episode.rewards), episode.rewards.shape[0]) for episode in batch])
        normalized_returns = (raw_returns - np.mean(raw_returns)) / (np.std(raw_returns) + np.finfo(np.float32).eps)
        return make_variable(normalized_returns)
