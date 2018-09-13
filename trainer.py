import torch
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader
from torch import no_grad

from datasets.viper import Mode


class Trainer:
    def __init__(self, name, model, data_set, optimizer, scheduler, criterion, plot, batch_size=64, max_epoch=50, log_interval=15):
        train, val, test = data_set
        self.batch_size = batch_size
        self.train_set = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        self.test_set = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
        self.validate_set = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)
        self.distance = PairwiseDistance()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.name = name
        self.model = model
        self.plot = plot
        self.max_epoch = max_epoch
        self.log_interval = log_interval
        self.best_accuracy = 0

    def train(self, epoch_id):
        self.model.train()

        summary_correct = 0
        summary_loss = 0

        for batch, inputs in enumerate(self.train_set):
            inputs = [i[1].cuda() for i in inputs]
            self.optimizer.zero_grad()
            correct, a, p, n = self.model(*inputs)
            loss = self.criterion(a, p, n)
            summary_loss += loss.item()
            summary_correct += correct.sum().item()
            loss.backward()
            self.optimizer.step()

            if batch % self.log_interval == 0:
                print('Epoch {} [{:5.0f}/{} ({:2.1f}%)] Loss: {:.6f} Accuracy: {:.1f}%'.format(
                    epoch_id, batch * len(self.train_set.dataset) / len(self.train_set), len(self.train_set.dataset),
                    100. * batch / len(self.train_set), loss.item(), 100. * correct.sum().item() / self.batch_size
                ))

        print('[TRAIN] Average loss: {:.6f}, Accuracy: {}/{} ({:2.1f}%)'.format(
            summary_loss / len(self.train_set.dataset), summary_correct, len(self.train_set.dataset),
            100. * summary_correct / len(self.train_set.dataset)
        ))

        self.plot(Mode.TRAIN, summary_loss / len(self.train_set.dataset),
                  100. * summary_correct / len(self.train_set.dataset))

    def validate(self):
        self.model.eval()

        summary_correct = 0
        loss = 0

        for inputs in self.validate_set:
            inputs = [i[1].cuda() for i in inputs]
            with no_grad():
                correct, anchors, positives, negatives = self.model(*inputs)
                loss += self.criterion(anchors, positives, negatives).item()
                summary_correct += correct.sum().item()

        if self.scheduler is not None:
            self.scheduler.step(loss / len(self.validate_set.dataset))

        avg_accuracy = 100. * summary_correct / len(self.validate_set.dataset)

        print('[VALIDATE] Average loss: {:.6f}, Accuracy: {}/{} ({:2.1f}%)'.format(
            loss / len(self.validate_set.dataset), summary_correct, len(self.validate_set.dataset), avg_accuracy))
        self.plot(Mode.VALIDATE, loss / len(self.validate_set.dataset), avg_accuracy)

        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            torch.save(self.model.state_dict(), self.name + '_best')

    def test(self, model):
        model.eval()
        ranks = {}

        for i, inputs in enumerate(self.test_set):
            print(100. * i / len(self.test_set))
            labels = inputs[0][0]
            inputs = [i[1].cuda() for i in inputs]
            with no_grad():
                correct, anchors, positives, negatives = model(*inputs)

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
            # self.validate()
            torch.save(self.model.state_dict(), self.name + '_last')
            print()

        # self.plot(Mode.TEST, self.test(self.model), None)
        # self.model.load_state_dict(torch.load(self.name + '_best'))
        # self.plot(Mode.TEST, self.test(self.model), None)
