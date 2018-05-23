import torch.nn as nn

from models.bkw import BkwNet


class DistNet(nn.Module):

    def __init__(self):
        super(DistNet, self).__init__()
        self.embedding_net = BkwNet()
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding_net(x)
        return self.fc(x)
