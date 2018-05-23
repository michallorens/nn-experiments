from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = f.relu(self.hidden(x))
        return self.output(x)


mlp = MLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.01)

inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]))
targets = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0],
    [1],
    [1],
    [0]
]))

for input, target in zip(inputs, targets):
    output = mlp(input)
    print(output)

#for param in mlp.parameters():
#    print(param.data)

for epoch in range(10000):
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()
        output = mlp(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #print('[epoch: %d] loss: %.3f' % (epoch, loss.item()))

#for param in mlp.parameters():
#    print(param.data)

for input, target in zip(inputs, targets):
    output = mlp(input)
    print(output)