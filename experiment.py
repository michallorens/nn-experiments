from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from datasets.viper import VIPeR
from models.inception import inception_v3
from models.inceptionmod import InceptionModNet
from models.resnet import ResNet
from models.triplet import TripletNet
from plotter import Plot
from trainer import Trainer


name = 'inception-triplet-128' + datetime.now().strftime('_%Y-%m-%d_%H%M%S')
plot = Plot(name)
kwargs = {'aux_logits': False}
net = TripletNet(InceptionModNet(1024, 128, **kwargs))
net.cuda()

criterion = nn.TripletMarginLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=3, eps=1e-8, verbose=True)

trainer = Trainer(name, net, VIPeR.create((316, 380), shuffle_seed=1), optimizer, scheduler, criterion, plot,
                  batch_size=2, log_interval=10, max_epoch=20)
trainer.run()
