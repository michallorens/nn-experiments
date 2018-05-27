import copy

import torch
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader
from torch import no_grad

from datasets.viper import Mode


class Trainer:
    def __init__(self, name, model, data_set, optimizer, scheduler, criterion, plot, batch_size=64, max_epoch=50, log_interval=15):
        train, val, test = data_set
        self.batch_size = batch_size

        self.train_set = tuple([DataLoader(t, batch_size=batch_size, drop_last=True) for t in train])
        self.train_batches = len(self.train_set[0])
        self.train_len = len(self.train_set[0].dataset)
        self.test_set = tuple([DataLoader(t, batch_size=batch_size, drop_last=True) for t in test])
        self.test_batches = len(self.test_set[0])
        self.validate_set = tuple([DataLoader(v, batch_size=batch_size, drop_last=True) for v in val])
        self.validate_len = len(self.validate_set[0].dataset)

        self.dist = PairwiseDistance()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.model = model
        self.plot = plot
        self.max_epoch = max_epoch
        self.log_interval = log_interval
        self.name = name
        self.best_accuracy = 0

    def train(self, epoch_id):
        self.model.train()

        summary_correct = 0
        summary_loss = 0

        for batch, inputs in enumerate(zip(*self.train_set)):
            self.optimizer.zero_grad()
            inputs = [[i[1].cuda() for i in j] for j in inputs]
            a1, a2, a3, p1, p2, p3, n1, n2, n3 = self.model(*inputs)
            loss1 = self.criterion(a1, p1, n1)
            loss2 = self.criterion(a2, p2, n2)
            loss3 = self.criterion(a3, p3, n3)
            loss = (loss1 + loss2 + loss3) / 3
            summary_loss += loss.item()
            a = (a1 + a2 + a3) / 3
            p = (p1 + p2 + p3) / 3
            n = (n1 + n2 + n3) / 3
            correct = (self.dist(a, p) < self.dist(a, n)).sum().item()
            summary_correct += correct
            loss.backward()
            self.optimizer.step()

            if batch % self.log_interval == 0:
                print('Epoch {} [{:5.0f}/{} ({:2.1f}%)] Loss: {:.6f} Accuracy: {:.1f}%'.format(
                    epoch_id, batch * self.train_len / self.train_batches, self.train_len,
                    100. * batch / self.train_batches, loss.item(), 100. * correct / self.batch_size
                ))

        print('[TRAIN] Average loss: {:.6f}, Accuracy: {}/{} ({:2.1f}%)'.format(
            summary_loss / self.train_len, summary_correct, self.train_len, 100. * summary_correct / self.train_len))

        self.plot(Mode.TRAIN, summary_loss / self.train_len, 100. * summary_correct / self.train_len)

    def validate(self):
        self.model.eval()

        summary_correct = 0
        loss = 0

        for inputs in zip(*self.validate_set):
            with no_grad():
                inputs = [[i[1].cuda() for i in j] for j in inputs]
                a1, a2, a3, p1, p2, p3, n1, n2, n3 = self.model(*inputs)
                loss1 = self.criterion(a1, p1, n1)
                loss2 = self.criterion(a2, p2, n2)
                loss3 = self.criterion(a3, p3, n3)
                loss += ((loss1 + loss2 + loss3) / 3).item()
                a = (a1 + a2 + a3) / 3
                p = (p1 + p2 + p3) / 3
                n = (n1 + n2 + n3) / 3
                correct = (self.dist(a, p) < self.dist(a, n)).sum().item()
                summary_correct += correct

        if self.scheduler is not None:
            self.scheduler.step(loss / self.validate_len)

        avg_accuracy = 100. * summary_correct / self.validate_len

        print('[VALIDATE] Average loss: {:.6f}, Accuracy: {}/{} ({:2.1f}%)'.format(
            loss / self.validate_len, summary_correct, self.validate_len, avg_accuracy))

        self.plot(Mode.VALIDATE, loss / self.validate_len, avg_accuracy)

        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            torch.save(self.model.state_dict(), self.name + '_best')

    def test(self, model):
        model.eval()
        ranks = {}

        for i, inputs in enumerate(zip(*self.test_set)):
            print(100. * i / self.test_batches)
            labels = inputs[0][0][0]
            inputs = [[i[1].cuda() for i in j] for j in inputs]
            with no_grad():
                a1, a2, a3, p1, p2, p3, n1, n2, n3 = self.model(*inputs)
                a = (a1 + a2 + a3) / 3
                p = (p1 + p2 + p3) / 3
                n = (n1 + n2 + n3) / 3
                correct = self.dist(a, p) < self.dist(a, n)
                # loss += self.criterion(anchors, positives, negatives).item()
            # index of highest probability
            for j, label in enumerate(labels):
                if label not in ranks.keys():
                    ranks[label] = 1
                else:
                    ranks[label] += 1 - correct[j].item()

        accu_rank = []
        for i in range(1, 101):
            accu_rank.append(100. * len([k for k in ranks.keys() if i >= ranks[k]]) / len(ranks.keys()))

        return accu_rank

    def run(self):
        for epoch in range(0, self.max_epoch):
            self.train(epoch)
            self.validate()
            torch.save(self.model.state_dict(), self.name + '_last')
            print()

        self.plot(Mode.TEST, self.test(self.model), None)
        self.model.load_state_dict(torch.load(self.name + '_best'))
        self.plot(Mode.TEST, self.test(self.model), None)
