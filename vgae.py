import torch
from torch_geometric.datasets import Planetoid, Amazon
# import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F
from torch_geometric.nn import GAE, VGAE
from eval import label_classification
from utils import set_seed





def edge_index_whole_layers(data):

    edge_index0 = data.edge_index
    tmp = torch.tensor([0, data.num_nodes])
    tmp = tmp.unsqueeze(1)
    mul_edge01 = edge_index0 + tmp

    edge_index1 = edge_index0 + data.num_nodes
    mul_edge12 = edge_index1 + tmp
    mul_edge = torch.cat((mul_edge01, mul_edge12), dim=1)

    edge_index2 = edge_index1 + data.num_nodes

    edge_index = torch.cat((edge_index0, edge_index1, edge_index2), dim=1)
    edge_self_I0 = torch.LongTensor([i for i in range(data.num_nodes)])
    edge_self_I1 = edge_self_I0 + data.num_nodes
    edge_self_I0 = edge_self_I0.unsqueeze(0)
    edge_self_I1 = edge_self_I1.unsqueeze(0)

    edge_self01 = torch.cat((edge_self_I0, edge_self_I1), dim=0)
    edge_self12 = edge_self01 + data.num_nodes
    edge_self = torch.cat((edge_self01, edge_self12), dim=1)

    edge_index = torch.cat((edge_index, edge_self,  mul_edge), dim=1)

    return edge_index

def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)

    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    adjacency_matrix[edge_index[1], edge_index[0]] = 1

    return adjacency_matrix

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # in_channels 是特征数量, out_channels * 2 是因为我们有两个GCNConv, 最后我们得到embedding大小的向量
        # cached 因为我们只有一张图
        self.conv1 = GCNConv(in_channels, out_channels * 2, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(out_channels * 2, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def incomplete_negative(data):
    data_edge = data.clone()
    data_edge.train_mask = data_edge.train_mask = data_edge.train_mask = None
    # data_edge = train_test_split_edges(data_edge, val_ratio=0.1, test_ratio=0.3)
    edge_split = RandomLinkSplit(split_labels=True)
    data_edge_train, data_edge_val, data_edge_test = edge_split(data_edge)

    out_channels = 128
    num_features = data.shape[1]
    epochs = 100

    model_pre = VGAE(VGCNEncoder(num_features, out_channels))  # new line
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_pre.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    train_pos_edge_index = data_edge_train.pos_edge_label_index
    train_pos_edge_index = train_pos_edge_index.to(device)
    test_pos_edge_index = data_edge_test.pos_edge_label_index
    test_pos_edge_index = test_pos_edge_index.to(device)
    test_neg_edge_index = data_edge_test.neg_edge_label_index
    test_neg_edge_index = test_neg_edge_index.to(device)
    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def vgae_train():
        model.train()
        optimizer.zero_grad()
        # model.encode 调用了我们传入的编码器
        # z = model.encode(x, train_pos_edge_index)
        # # recon_loss 为重构损失
        # loss = model.recon_loss(z, train_pos_edge_index)

        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        # vgae
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)

    # start training
    for epoch in range(1, epochs + 1):
        vgae_train()

    A = edge_index_to_adjacency_matrix(edge_index, data.num_nodes).to(device)
    A = A.float()
    I = torch.eye(data.num_nodes).to(device)
    D = torch.sum(A, dim=1)
    D = D.unsqueeze(1)

    z = model.encode(x, edge_index)
    z1 = (A @ z) / (D + 1) + z
    z2 = (A @ z1) / (D + 1) + z
    z = F.normalize(z)
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    one = torch.ones(data.num_nodes, data.num_nodes).to(device)

    Sim02 = torch.mm(z, z2.t())
    Sim02 = (Sim02 + one) / 2

    Sim12 = torch.mm(z1, z2.t())
    Sim12 = (Sim12 + one) / 2

    Sim22 = torch.mm(z2, z2.t())
    Sim22 = (Sim22 + one) / 2

    Sim20 = Sim02.t()
    Sim21 = Sim12.t()

    One = torch.ones((data.num_nodes, data.num_nodes)).to("cuda")
    DS02 = One - Sim02
    DS12 = One - Sim12
    DS22 = One - Sim22
    DS21 = One - Sim21
    DS20 = One - Sim20


    diagonal_elements = torch.diag(DS02)
    res_DS02 = DS02 - torch.diag(diagonal_elements)

    diagonal_elements = torch.diag(DS12)
    res_DS12 = DS12 - torch.diag(diagonal_elements)

    diagonal_elements = torch.diag(DS22)
    res_DS22 = DS22 - torch.diag(diagonal_elements)

    diagonal_elements = torch.diag(DS21)
    res_DS21 = DS21 - torch.diag(diagonal_elements)

    diagonal_elements = torch.diag(DS20)
    res_DS20 = DS20 - torch.diag(diagonal_elements)
    return [res_DS02, res_DS20, res_DS12, res_DS21, res_DS22]



