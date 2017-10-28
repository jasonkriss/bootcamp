import numpy as np

from bootcamp.utils import compute_return

class AverageReturn(object):
    def __call__(self, state):
        return np.mean([compute_return(episode.rewards) for episode in state['batch']])
