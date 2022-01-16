import torch.nn as nn


# Taken from https://github.com/team-approx-bayes/fromp/blob/main/models.py
# Fully connected network, size: input_size, hidden_size, ..., output_size
class MLP(nn.Module):
    def __init__(self, size, act='sigmoid'):
        super(type(self), self).__init__()
        self.num_layers = len(size) - 1
        lower_modules = []
        for i in range(self.num_layers - 1):
            lower_modules.append(nn.Linear(size[i], size[i+1]))
            if act == 'relu':
                lower_modules.append(nn.ReLU())
            elif act == 'sigmoid':
                lower_modules.append(nn.Sigmoid())
            else:
                raise ValueError(f"{act} activation hasn't been implemented")
        self.layer_1 = nn.Sequential(*lower_modules)
        self.layer_2 = nn.Linear(size[-2], size[-1])

    def forward(self, x):
        o = self.layer_1(x)
        o = self.layer_2(o)
        return o
