from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torch import no_grad
from torch.utils.data import DataLoader

from datasets.viper import VIPeR, Mode
from metric.kissme import KISSME
from models.bkw import BkwNet
from models.triplet import TripletNet
from plotter import Plot
from trainer import Trainer

name = 'shallow-triplet-kissme' + datetime.now().strftime('_%Y-%m-%d_%H%M%S')
plot = Plot(name)
net = TripletNet(BkwNet())
net.load_state_dict(torch.load('shallow-triplet-64_2018-05-25_202105_model'))
net.cuda()

train, _, _ = VIPeR.create((316, 380), negative_samples=1)
train_loader = DataLoader(train, batch_size=64, shuffle=False, pin_memory=False, drop_last=False)
features = []
identities = []

for inputs in train_loader:
    a_labels = inputs[0][0]
    p_labels = inputs[1][0]
    inputs = [i[1].cuda() for i in inputs]
    with no_grad():
        anchors, positives, _ = net(*inputs)
    anchors = anchors.data.cpu()
    positives = positives.data.cpu()

    for output, label in zip(anchors, a_labels):
        identities.append(label)
        features.append(output)

    for output, label in zip(positives, p_labels):
        identities.append(label)
        features.append(output)

features = torch.stack(features).numpy()
identities = list(map(int, identities))
identities = torch.Tensor(identities).numpy()
metric = KISSME()
metric.fit(features, identities)

criterion = nn.TripletMarginLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=1, eps=1e-8, verbose=True)

trainer = Trainer(name, net, VIPeR.create((316, 380)), optimizer, scheduler, criterion, plot,
                  batch_size=64, log_interval=10, max_epoch=15, metric=metric)
plot(Mode.TEST, trainer.test(net), None)
