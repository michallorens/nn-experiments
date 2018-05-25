from __future__ import absolute_import

from torch import nn
from torch.nn import functional as f
from torch.nn import init
import torchvision


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, fc1_dim, fc2_dim, pre_trained=True):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pre_trained = pre_trained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=self.pre_trained)

        self.fc1 = nn.Linear(self.base.fc.in_features, fc1_dim)
        self.bn = nn.BatchNorm1d(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        init.kaiming_normal(self.fc1.weight, mode='fan_out')
        init.kaiming_normal(self.fc2.weight, mode='fan_out')
        init.constant(self.fc1.bias, 0)
        init.constant(self.fc2.bias, 0)
        init.constant(self.bn.weight, 1)
        init.constant(self.bn.bias, 0)

        if not self.pre_trained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = f.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn(x)
        x = f.relu(x)
        x = f.dropout(x, 0.5)
        return self.fc2(x)

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
