import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers import Conv2dReparameterization, Conv2dFlipout
from bayesian_torch.layers import LinearReparameterization, LinearFlipout


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, var0=1, dropRate=0.0, estimator='reparam'):
        _check_estimator(estimator)
        super().__init__()

        Conv2dVB = Conv2dReparameterization if estimator == 'reparam' else Conv2dFlipout

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2dVB(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
            prior_variance=var0
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2dVB(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False,
            prior_variance=var0
        )
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        kl_total = 0

        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        if self.equalInOut:
            out, kl = self.conv1(out)
            out = self.relu2(self.bn2(out))
        else:
            out, kl = self.conv1(x)
            out = self.relu2(self.bn2(out))

        kl_total += kl

        out, kl = self.conv2(out)
        kl_total += kl

        if not self.equalInOut:
            out_shortcut = self.convShortcut(x)
            return torch.add(out_shortcut, out), kl_total
        else:
            return torch.add(x, out), kl_total


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, var0=1, dropRate=0.0, estimator='reparam'):
        _check_estimator(estimator)
        super().__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, var0, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, var0, dropRate):
        layers = []

        for i in range(nb_layers):
            layers.append(block(
                i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                var0, dropRate
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        kl_total = 0

        for l in self.layer:
            out, kl = l(out)
            kl_total += kl

        return out, kl_total


class WideResNetBBB(nn.Module):

    def __init__(self, depth, widen_factor, num_classes, num_channel=3, var0=1, droprate=0, estimator='reparam', feature_extractor=False):
        _check_estimator(estimator)
        super().__init__()

        Conv2dVB = Conv2dReparameterization if estimator == 'reparam' else Conv2dFlipout
        LinearVB = LinearReparameterization if estimator == 'reparam' else LinearFlipout

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = Conv2dVB(
            num_channel, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False,
            prior_variance=var0
        )
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, var0, droprate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, var0, droprate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, var0, droprate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = LinearVB(nChannels[3], num_classes, prior_variance=var0)

        self.nChannels = nChannels[3]
        self.feature_extractor = feature_extractor

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out, kl_total = self.features(x)

        if self.feature_extractor:
            return out, kl_total

        out, kl = self.fc(out)
        kl_total += kl

        return out, kl_total


    def features(self, x):
        kl_total = 0

        out, kl = self.conv1(x)
        kl_total += kl
        out, kl = self.block1(out)
        kl_total += kl
        out, kl = self.block2(out)
        kl_total += kl
        out, kl = self.block3(out)
        kl_total += kl

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        return out, kl_total


def _check_estimator(estimator):
    assert estimator in ['reparam', 'flipout'], 'Estimator must be either "reparam" or "flipout"'
