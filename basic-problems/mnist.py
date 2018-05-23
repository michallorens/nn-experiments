import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

kwargs = {'pin_memory': True}
batch_size = 256
log_interval = 50
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = MNIST('./data', train=True, download=True, transform=transform)
mnist_test = MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(mnist_test, batch_size, shuffle=True, **kwargs)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.dropout = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = f.relu(self.pool(self.conv1(x)))
        x = f.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        #x = f.dropout(x, training=self.training)
        return f.log_softmax(self.fc2(x), dim=1)


net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


def train(epoch):
    net.train()
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = net(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_id % log_interval == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item()
            ))


def test():
    net.eval()
    loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = net(data)
        loss += f.nll_loss(output, target).item()
        #index of highest probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    loss /= len(test_loader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


test()

for epoch in range(0, 20):
    train(epoch)
    test()
