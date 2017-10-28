from torch.nn import Sequential, Linear, ReLU, Module

class MLP(Module):
    def __init__(self, in_features, out_features, hidden_features=[]):
        super(MLP, self).__init__()
        self.sequential = Sequential()
        for i, features in enumerate(hidden_features):
            self.sequential.add_module('hidden-{}-linear'.format(i), Linear(in_features, features))
            self.sequential.add_module('hidden-{}-activation'.format(i), ReLU())
            in_features = features
        self.sequential.add_module('final', Linear(in_features, out_features))

    def forward(self, x):
        return self.sequential(x)
