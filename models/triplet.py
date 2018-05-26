from torch import nn


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        a = self.embedding_net(anchor)
        p = self.embedding_net(positive)
        n = self.embedding_net(negative)

        return a, p, n
