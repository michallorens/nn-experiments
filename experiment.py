from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from torchvision import transforms

from datasets.cuhk01 import CUHK01
from datasets.transforms.transforms import Scale, Slice
from datasets.viper import VIPeR
from models.bkw import BkwNet
from models.convnet import ConvNet
from models.distnet import DistNet
from models.parts import PartsNet
from models.triplet import TripletNet
from models.voting import VotingNet
from plotter import Plot
from trainer import Trainer

name = 'triplet-sgd-distnet' + datetime.now().strftime('_%Y-%m-%d_%H%M%S')
plot = Plot(name)
net = TripletNet(BkwNet())
# net = VotingNet(BkwNet())
net.cuda()

criterion = nn.TripletMarginLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=1e-6)
scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=1, eps=1e-8, verbose=True)

trainer = Trainer(name, net, VIPeR.create((316, 380)), optimizer, scheduler, criterion, plot,
                  batch_size=64, log_interval=10, max_epoch=15)
trainer.run()
