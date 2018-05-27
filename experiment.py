from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from torchvision import transforms

from datasets.transforms.transforms import Scale, Slice
from datasets.viper import VIPeR
from models.bkw import BkwNet
from models.parts import PartsNet
from trainer import Trainer
from plotter import Plot


name = 'triplet-sgd-parts' + datetime.now().strftime('_%Y-%m-%d_%H%M%S')
plot = Plot(name)
net = PartsNet(BkwNet())
# net = VotingNet(BkwNet())
net.cuda()

criterion = nn.TripletMarginLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=1e-6)
scheduler = scheduler.ReduceLROnPlateau(optimizer, patience=1, eps=1e-8, verbose=True)

splits = (316, 380)
seed = 12345

whole_train, whole_val, whole_test = VIPeR.create(splits, shuffle_seed=seed)
top_train, top_val, top_test = VIPeR.create(splits, shuffle_seed=seed,
                                            train_transform=transforms.Compose([
                                                Slice(2, 1),
                                                transforms.RandomSizedCrop(64),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                            ]),
                                            transform=transforms.Compose([
                                                Slice(2, 1),
                                                Scale(64),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                            ]))
bot_train, bot_val, bot_test = VIPeR.create(splits, shuffle_seed=seed,
                                            train_transform=transforms.Compose([
                                                Slice(2, 2),
                                                transforms.RandomSizedCrop(64),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                            ]),
                                            transform=transforms.Compose([
                                                Slice(2, 2),
                                                Scale(64),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                            ]))

train = (whole_train, top_train, bot_train)
val = (whole_val, top_val, bot_val)
test = (whole_test, top_test, bot_test)

trainer = Trainer(name, net, (train, val, test), optimizer, scheduler, criterion, plot,
                  batch_size=64, log_interval=10, max_epoch=30)
trainer.run()
