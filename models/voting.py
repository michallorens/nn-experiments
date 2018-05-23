from torch import nn

from models.triplet import TripletNet


class VotingNet(nn.Module):
    def __init__(self, embedding_net):
        super(VotingNet, self).__init__()
        self.triplet_net1 = TripletNet(embedding_net)
        self.triplet_net2 = TripletNet(embedding_net)
        self.triplet_net3 = TripletNet(embedding_net)
        self.distance = nn.PairwiseDistance()

    def forward(self, anchor, positive, negative):
        _, a1, p1, n1 = self.triplet_net1(anchor, positive, negative)
        _, a2, p2, n2 = self.triplet_net2(anchor, positive, negative)
        _, a3, p3, n3 = self.triplet_net3(anchor, positive, negative)

        a = (a1 + a2 + a3) / 3
        p = (p1 + p2 + p3) / 3
        n = (n1 + n2 + n3) / 3

        return self.distance(a, p) < self.distance(a, n), a, p, n
