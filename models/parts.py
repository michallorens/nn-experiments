from torch import nn

from models.triplet import TripletNet


class PartsNet(nn.Module):
    def __init__(self, embedding_net):
        super(PartsNet, self).__init__()
        self.triplet_net1 = TripletNet(embedding_net)
        self.triplet_net2 = TripletNet(embedding_net)
        self.triplet_net3 = TripletNet(embedding_net)

    def forward(self, whole, top, bottom):
        _, a1, p1, n1 = self.triplet_net1(*whole)
        _, a2, p2, n2 = self.triplet_net2(*top)
        _, a3, p3, n3 = self.triplet_net3(*bottom)

        return a1, a2, a3, p1, p2, p3, n1, n2, n3
