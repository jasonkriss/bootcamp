class Agent(object):
    def train(self, batch):
        raise NotImplementedError

    def act(self, observation):
        raise NotImplementedError
