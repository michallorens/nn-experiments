from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from datasets.viper import VIPeR
from models.inception import inception_v3
from models.resnet import ResNet
from models.triplet import TripletNet
from plotter import Plot
from trainer import Trainer


name = 'inception-triplet-64' + datetime.now().strftime('_%Y-%m-%d_%H%M%S')
plot = Plot(name)
kwargs = {'num_classes': 64}
net = TripletNet(inception_v3(**kwargs))
net.cuda()

criterion = nn.TripletMarginLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=1, eps=1e-8, verbose=True)

trainer = Trainer(name, net, VIPeR.create((316, 380)), optimizer, scheduler, criterion, plot,
                  batch_size=64, log_interval=10, max_epoch=15)
trainer.run()
