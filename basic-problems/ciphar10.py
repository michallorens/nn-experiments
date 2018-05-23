import plotly.offline as po
from plotly import tools
from plotly.graph_objs import Scatter, Layout

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


kwargs = {'pin_memory': True}
max_epoch = 50
batch_size = 32
log_interval = 50
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4734,), (0.2515,))
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
test_set = CIFAR10('./data/cifar10', train=False, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_set, batch_size, shuffle=True, **kwargs)

train_loss = []
test_loss = []
train_accu = []
test_accu = []
epochs = list(range(0, max_epoch))
auto_open = True


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.2)
        self.norm1 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 40, 3)
        self.conv4 = nn.Conv2d(40, 40, 3)
        self.norm2 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 250)
        self.norm3 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.pool(self.conv2(x)))
        x = self.drop(self.norm1(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.pool(self.conv4(x)))
        x = self.drop(self.norm2(x))
        x = x.view(-1, 1000)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.dropout(self.norm3(x), 0.5)
        x = f.relu(self.fc3(x))
        return self.fc4(x)


net = Net()
net.cuda()
criterion = nn.TripletMarginLoss.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=1e-6)


def train(epoch_id):
    net.train()
    running_loss = 0
    correct = 0

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (output.max(dim=1)[1] == target).sum().item()

        if batch_id % log_interval == 0:
            print('Epoch {} [{:5d}/{} ({:2.1f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(
                epoch_id, batch_id * len(data), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item(),
                100. * (output.max(dim=1)[1] == target).sum().float() / target.size()[0]
            ))

    running_loss /= len(train_loader.dataset)
    correct_pct = 100. * correct / len(train_loader.dataset)

    print('[TRAIN] Average loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.format(
        running_loss, correct, len(train_loader.dataset), correct_pct
    ))

    train_loss.append(running_loss)
    train_accu.append(correct_pct)


def test():
    net.eval()
    loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = net(data)
        loss += criterion(output, target).item()
        #index of highest probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    loss /= len(test_loader.dataset)
    correct_pct = 100. * correct.float() / len(test_loader.dataset)

    print('[TEST] Average loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)'.format(
        loss, correct, len(test_loader.dataset), correct_pct
    ))

    test_loss.append(loss)
    test_accu.append(correct_pct)


def plot():
    tr_acc = Scatter(x=epochs, y=train_accu, mode='lines', name='train accuracy')
    ts_acc = Scatter(x=epochs, y=test_accu, mode='lines', name='test accuracy')

    tr_loss = Scatter(x=epochs, y=train_loss, mode='lines', name='train loss')
    ts_loss = Scatter(x=epochs, y=test_loss, mode='lines', name='test loss')

    chart = tools.make_subplots(rows=2, cols=1, subplot_titles=('Loss', 'Accuracy'), print_grid=False)
    chart.append_trace(tr_loss, 1, 1)
    chart.append_trace(ts_loss, 1, 1)
    chart.append_trace(tr_acc, 2, 1)
    chart.append_trace(ts_acc, 2, 1)

    chart['layout']['xaxis1'].update(title='Epoch', range=[0, max_epoch])
    chart['layout']['xaxis2'].update(title='Epoch', range=[0, max_epoch])

    chart['layout']['yaxis1'].update(title='Loss')
    chart['layout']['yaxis2'].update(title='Accuracy')

    chart['layout'].update(title='CIPHAR-10')

    po.plot(chart, filename='ciphar-10.html', auto_open=auto_open)


for epoch in range(0, max_epoch):
    train(epoch)
    test()
    plot()
    auto_open = False
    print()
