import torch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import Conv2dReparameterization, Conv2dFlipout
from bayesian_torch.layers import LinearReparameterization, LinearFlipout


class LeNetBBB(nn.Module):

    def __init__(self, num_classes=10, var0=1, estimator='flipout'):
        _check_estimator(estimator)
        super().__init__()

        Conv2dVB = Conv2dReparameterization if estimator == 'reparam' else Conv2dFlipout
        LinearVB = LinearReparameterization if estimator == 'reparam' else LinearFlipout

        self.conv1 = Conv2dVB(1, 6, 5, prior_variance=var0)
        self.conv2 = Conv2dVB(6, 16, 5, prior_variance=var0)
        self.flatten = nn.Flatten()
        self.fc1 = LinearVB(256, 120, prior_variance=var0)
        self.fc2 = LinearVB(120, 84, prior_variance=var0)
        self.fc3 = LinearVB(84, num_classes, prior_variance=var0)

    def forward(self, x):
        x, kl_total = self.features(x)
        x, kl = self.fc3(x)
        kl_total += kl

        return x, kl_total


    def features(self, x, return_acts=False):
        kl_total = 0

        x, kl = self.conv1(x)
        kl_total += kl
        x = F.max_pool2d(F.relu(x), 2, 2)

        x, kl = self.conv2(x)
        kl_total += kl
        x = F.max_pool2d(F.relu(x), 2, 2)

        x = self.flatten(x)

        x, kl = self.fc1(x)
        kl_total += kl
        x = F.relu(x)

        x, kl = self.fc2(x)
        kl_total += kl
        x = F.relu(x)

        return x, kl_total


def _check_estimator(estimator):
    assert estimator in ['reparam', 'flipout'], 'Estimator must be either "reparam" or "flipout"'
