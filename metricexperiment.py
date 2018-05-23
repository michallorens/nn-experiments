from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from metric_learn import ITML_Supervised

from torchvision import transforms

from datasets.cuhk01 import CUHK01
from datasets.transforms.transforms import Scale, Slice
from datasets.viper import VIPeR
from metric.metric import metric_learn
from models.bkw import BkwNet
from models.convnet import ConvNet
from models.parts import PartsNet
from models.triplet import TripletNet
from models.voting import VotingNet
from plotter import Plot
from trainer import Trainer

name = 'triplet-sgd-nopool' + datetime.now().strftime('_%Y-%m-%d_%H%M%S')
plot = Plot(name)
net = TripletNet(ConvNet())
net.load_state_dict(torch.load('triplet-sgd_2018-05-10_101323_best'))
# net = VotingNet(BkwNet())
net.cuda()

train, val, test = VIPeR.create((316, 380), shuffle_seed=12345)

metric = ITML_Supervised()
metric_learn(metric, net, train)

criterion = nn.TripletMarginLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=1, eps=1e-8, verbose=True)

trainer = Trainer(name, net, (train, val, test), optimizer, scheduler, criterion, plot,
                  batch_size=64, log_interval=10, max_epoch=15)
trainer.test(net)
