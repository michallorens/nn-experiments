import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init

from models.inception import inception_v3


class InceptionModNet(nn.Module):

    def __init__(self, fc1_dim, fc2_dim, pretrained=False, **kwargs):
        super(InceptionModNet, self).__init__()

        self.base = inception_v3(pretrained=pretrained, **kwargs)

        self.fc1 = nn.Linear(self.base.fc.in_features, fc1_dim)
        self.bn = nn.BatchNorm1d(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        init.kaiming_normal(self.fc1.weight, mode='fan_out')
        init.kaiming_normal(self.fc2.weight, mode='fan_out')
        init.constant(self.fc1.bias, 0)
        init.constant(self.fc2.bias, 0)
        init.constant(self.bn.weight, 1)
        init.constant(self.bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avg_pool2d':
                break
            x = module(x)

        x = f.avg_pool2d(x, kernel_size=8)
        x = f.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn(x)
        x = f.relu(x)
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

