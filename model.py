import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Multi-layer(2-layer) Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)

# Multi-layer Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers=2):
        super(GCN, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.fc = nn.Linear(out_dim * 2, out_dim)

        self.convs.append(GCNConv(in_dim, out_dim * 2))
        for _ in range(self.num_layers - 2):
            self.convs.append(GCNConv(out_dim * 2, out_dim * 2))

        self.convs.append(GCNConv(out_dim * 2, out_dim))
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x0 = x
        for i in range(self.num_layers):
            x = self.act_fn(self.convs[i](x, edge_index))
            x0 = self.convs[i].lin(x0)
            if i == 0:
                h1 = self.convs[1].lin(x)
                # if self.convs[1].bias is not None:
                #     h += self.convs[1].bias
                h1 = self.act_fn(h1)
            elif i < self.num_layers - 1:
                h = self.convs[i+1].lin(h)
                # if self.convs[i].bias is not None:
                #     h += self.convs[i].bias
                h1 = self.act_fn(h1)

        return x, x0, h1


class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp, DS, lambda0, lambda2):
        super(Model, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)
        self.DS = DS
        self.lam0 = lambda0
        self.lam2 = lambda2
    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s
    def get_loss_0k(self, z1, z2, h0):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        positive = f(self.sim(h0, z2)).diag()  # positive pairs
        refl_sim = torch.mul(f(self.sim(h0, z1)), self.DS[0])
        between_sim = torch.mul(f(self.sim(h0, z2)), self.DS[0])  # inter-view pairs  不同视图的其他节点和自己作为正例  如果Z1是1-layer Z2是3-layer

        x1 = refl_sim.sum(1) + between_sim.sum(1) + positive
        loss = -th.log(positive / x1)

        return loss

    def get_loss_k0(self, z1, z2, h0):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        positive = f(self.sim(h0, z2)).diag()  # positive pairs
        refl_sim = torch.mul(f(self.sim(h0, z1)), self.DS[1])
        between_sim = torch.mul(f(self.sim(h0, z2)), self.DS[1])  # inter-view pairs  不同视图的其他节点和自己作为正例  如果Z1是1-layer Z2是3-layer

        x1 = refl_sim.sum(1) + between_sim.sum(1) + positive
        loss = -th.log(positive / x1)

        return loss

    def get_loss_k1(self, z1, z2, h1):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)


        positive = f(self.sim(h1, z2)).diag()  # positive pairs
        refl_sim = torch.mul(f(self.sim(h1, z1)), self.DS[3])
        between_sim = torch.mul(f(self.sim(h1, z2)), self.DS[3])

        x1 = refl_sim.sum(1) + between_sim.sum(1) + positive
        loss = -th.log(positive / x1)

        return loss

    def get_loss_1k(self, z1, z2, h1):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)


        positive = f(self.sim(h1, z2)).diag()  # positive pairs
        refl_sim = torch.mul(f(self.sim(h1, z1)), self.DS[2])
        between_sim = torch.mul(f(self.sim(h1, z2)), self.DS[2])

        x1 = refl_sim.sum(1) + between_sim.sum(1) + positive
        loss = -th.log(positive / x1)

        return loss


    def get_loss_kk(self, z1, z2):
        # calculate SimCLR loss
        f = lambda x: th.exp(x / self.temp)

        positive = f(self.sim(z1, z2)).diag()  # positive pairs
        refl_sim = torch.mul(f(self.sim(z1, z1)), self.DS[4])
        between_sim = torch.mul(f(self.sim(z1, z2)), self.DS[4])  # inter-view pairs  不同视图的其他节点和自己作为正例  如果Z1是1-layer Z2是3-layer

        x1 = refl_sim.sum(1) + between_sim.sum(1) + positive
        loss = -th.log(positive / x1)

        return loss


    def get_embedding(self, feat, edge_index: torch.Tensor):
        # get embeddings from the model for evaluation
        x, _, y = self.encoder(feat, edge_index)
        # x = self.proj(x)
        # +MLP

        return x.detach()

    def forward(self, edge_index1: torch.Tensor, edge_index2: torch.Tensor, feat1, feat2) -> torch.Tensor:

        # encoding
        h, h0, h1 = self.encoder(feat1, edge_index1)
        H, H0, H1 = self.encoder(feat2, edge_index2)

        # projection
        z_h = self.proj(h)          # view1, K-layer representation
        z_H = self.proj(H)          # view2, K-layer representation
        z_h0 = self.proj(h0)        # view1, 0-layer representation
        z_H0 = self.proj(H0)        # view2, 0-layer representation
        z_h1 = self.proj(h1)        # view1, 1-layer representation
        z_H1 = self.proj(H1)        # view2, 1-layer representation


        # get loss
        l02_h = self.get_loss_k0(z_h, z_H, z_h0)         # loss0-k
        l20_H = self.get_loss_0k(z_H, z_h, z_H0)         # lossk-0
        l12_h = self.get_loss_k1(z_h, z_H, z_h1)         # loss1-k
        l21_H = self.get_loss_1k(z_H, z_h, z_H1)         # lossk-1
        l22_h = self.get_loss_kk(z_h, z_H)                # lossk-k
        l22_H = self.get_loss_kk(z_H, z_h)                # lossk-k

        l02 = (l02_h + l20_H) / 2
        l12 = (l12_h + l21_H) / 2
        l22 = (l22_h + l22_H) / 2

        ret = self.lam0*l02 + self.lam2*l22 + (1-self.lam0-self.lam2)*l12

        return ret.mean()

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret



