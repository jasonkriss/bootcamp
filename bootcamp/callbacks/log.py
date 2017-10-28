class Log(object):
    def __init__(self, logger, map_fn):
        self._logger = logger
        self._map_fn = map_fn

    def __call__(self, trainer, state):
        self._logger.log(state['iteration'], self._map_fn(state))
