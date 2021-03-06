from torch import nn


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.distance = nn.PairwiseDistance()

    def forward(self, anchor, positive, negative):
        a = self.embedding_net(anchor)
        p = self.embedding_net(positive)
        n = self.embedding_net(negative)

        return self.distance(a, p) < self.distance(a, n), a, p, n
