import numpy as np

from bootcamp.exceptions import StopTraining
from bootcamp.utils import compute_return

class Monitor(object):
    def __init__(self):
        self._returns = []

    def __call__(self, trainer, state):
        batch_returns = [compute_return(episode.rewards) for episode in state['batch']]
        self._returns += batch_returns
        relevant_returns = self._returns[-trainer.env.spec.trials:]
        average_return = np.mean(relevant_returns)
        if average_return > trainer.env.spec.reward_threshold:
            print('Solved after {} iterations'.format(state['iteration']))
            raise StopTraining
