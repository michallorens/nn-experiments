from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from datasets.viper import VIPeR
from models.bkw import BkwNet
from models.triplet import TripletNet
from plotter import Plot
from trainer import Trainer

name = 'shallow-triplet-128-bn' + datetime.now().strftime('_%Y-%m-%d_%H%M%S')
plot = Plot(name)
net = TripletNet(BkwNet(128))
net.cuda()

criterion = nn.TripletMarginLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=1, eps=1e-8, verbose=True)

trainer = Trainer(name, net, VIPeR.create((316, 380), shuffle_seed=1), optimizer, scheduler, criterion, plot,
                  batch_size=64, log_interval=10, max_epoch=30)
trainer.run()
