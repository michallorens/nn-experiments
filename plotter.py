import plotly.offline as po
from plotly import tools
from plotly.graph_objs import Scatter, Layout

from datasets.viper import Mode


class Plot:
    def __init__(self, title, epoch=50):
        self.title = title
        self.epochs = list(range(0, epoch))
        self.rank = list(range(0, 100))

        self.train_loss = []
        self.val_loss = []
        self.train_accu = []
        self.val_accu = []
        self.last_accu = []
        self.best_accu = []

        #self.auto_open = True
        self.auto_open = False
        self.test_toggle = False

    def __call__(self, mode, loss, correct):
        if mode == Mode.TRAIN:
            self.train_loss.append(loss)
            self.train_accu.append(correct)
        elif mode == Mode.VALIDATE:
            self.val_loss.append(loss)
            self.val_accu.append(correct)
        elif mode == Mode.TEST and self.test_toggle is False:
            self.last_accu = loss
            self.test_toggle = False
        elif mode == Mode.TEST and self.test_toggle is True:
            self.best_accu = loss
            self.test_toggle = True

        self.plot()
        self.auto_open = False

    def plot(self):
        tr_acc = Scatter(x=self.epochs, y=self.train_accu, mode='lines', name='Train accuracy')
        ts_acc = Scatter(x=self.epochs, y=self.val_accu, mode='lines', name='Validation accuracy')

        tr_loss = Scatter(x=self.epochs, y=self.train_loss, mode='lines', name='Train loss')
        ts_loss = Scatter(x=self.epochs, y=self.val_loss, mode='lines', name='Validation loss')

        last_accu = Scatter(x=self.rank, y=self.last_accu, mode='lines', name='CMC curve')

        best_accu = Scatter(x=self.rank, y=self.best_accu, mode='lines', name='CMC curve')

        chart = tools.make_subplots(rows=2, cols=2, subplot_titles=(
            'Loss', 'CMC Score after last epoch', 'Accuracy', 'CMC Score for best model'), print_grid=False)
        chart.append_trace(tr_loss, 1, 1)
        chart.append_trace(ts_loss, 1, 1)
        chart.append_trace(tr_acc, 2, 1)
        chart.append_trace(ts_acc, 2, 1)
        chart.append_trace(last_accu, 1, 2)
        chart.append_trace(best_accu, 2, 2)

        chart['layout']['xaxis1'].update(title='Epoch', range=[0, len(self.epochs)])
        chart['layout']['xaxis2'].update(title='Rank', range=[0, 100])
        chart['layout']['xaxis3'].update(title='Epoch', range=[0, len(self.epochs)])
        chart['layout']['xaxis4'].update(title='Rank', range=[0, 100])
        chart['layout']['yaxis1'].update(title='Loss')
        chart['layout']['yaxis2'].update(title='CMC Score')
        chart['layout']['yaxis3'].update(title='Accuracy')
        chart['layout']['yaxis4'].update(title='CMC Score')

        chart['layout'].update(title=self.title)

        po.plot(chart, filename=self.title + '.html', auto_open=self.auto_open)
