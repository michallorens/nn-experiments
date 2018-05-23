import torch.nn as nn
import torch.nn.functional as f


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 64)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = x.view(-1, 8192)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, 0.5)
        return self.fc2(x)
