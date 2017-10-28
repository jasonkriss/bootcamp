class Evaluate(object):
    STATE_KEY = 'metrics'

    def __init__(self, metrics):
        self._metrics = metrics

    def __call__(self, trainer, state):
        state[self.STATE_KEY] = {}
        for name, metric in self._metrics.items():
            state[self.STATE_KEY][name] = metric(state)
