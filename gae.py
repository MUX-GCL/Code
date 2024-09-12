import torch
from torch_geometric.datasets import Planetoid, Amazon
# import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F

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


if __name__ == '__main__':

    from torch_geometric.nn import GAE, VGAE

    # set random seed
    seed = 41
    set_seed(seed)

    # data
    # dataset = Planetoid(root='/data/Cora', name='Cora')
    # dataset = Amazon(root='/data/amazon-photo', name='Photo')
    # dataset = Planetoid(root='/data/CiteSeer', name='CiteSeer')
    dataset = Planetoid(root='/data/PubMed', name='PubMed')
    # dataset = Amazon(root='/data/amazon-computer', name='Computers')
    data = dataset[0]
    data_edge = data.clone()
    data_edge.train_mask = data_edge.train_mask = data_edge.train_mask = None
    # data_edge = train_test_split_edges(data_edge, val_ratio=0.1, test_ratio=0.3)
    edge_split = RandomLinkSplit(split_labels=True)
    data_edge_train, data_edge_val, data_edge_test = edge_split(data_edge)
    print(data)
    print(data_edge_train)
    print(data_edge_test)
    print(data_edge_val)


    # parameters
    out_channels = 128
    num_features = dataset.num_features
    epochs = 100

    # model
    # model = GAE(GCNEncoder(num_features, out_channels))
    # print(model)
    model = VGAE(VGCNEncoder(num_features, out_channels))  # new line
    print(model)

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # pos & neg edges
    # train_pos_edge_index = data_edge_train.edge_label_index[:, data_edge_train.edge_label.bool()]
    train_pos_edge_index = data_edge_train.pos_edge_label_index
    train_pos_edge_index = train_pos_edge_index.to(device)
    # test_pos_edge_index = data_edge_test.edge_label_index[:, data_edge_test.edge_label.bool()]
    test_pos_edge_index = data_edge_test.pos_edge_label_index
    test_pos_edge_index = test_pos_edge_index.to(device)
    # test_neg_edge_index = data_edge_test.edge_label_index[:, (1 - data_edge_test.edge_label).bool()]
    test_neg_edge_index = data_edge_test.neg_edge_label_index
    test_neg_edge_index = test_neg_edge_index.to(device)

    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def train():
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

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        # 使用正边和负边来测试模型的准确率
        return model.test(z, pos_edge_index, neg_edge_index)

    # start training
    # for epoch in range(1, epochs + 1):
    #     loss = train()
    #     auc, ap = test(test_pos_edge_index, test_neg_edge_index)
    #     print('Epoch: {:03d}, LOSS: {:.4f}'.format(epoch, loss))
    #     print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    #     model.eval()
    # with torch.no_grad():
    #     z = model.encode(x, edge_index)
    # label_classification(z, data.y, ratio=0.1)


    # with torch.no_grad():
    #     model.eval()
    #     torch.save(model.state_dict(), 'save_vgae_cora.pth')
    #     # z = model.encode(x, edge_index)
    #     # torch.save(z, "cora_vgae_vec.pt")
    #     # z = F.normalize(z)            # sigmoid  截断  前K高的SIm作为正例
    #     # sim = torch.mm(z, z.t())
    #     # sim = torch.clamp(sim, min=0)
    #     # torch.save(sim, "cora_gae_sim.pt")
    #     # print(z)
    A = edge_index_to_adjacency_matrix(edge_index, data.num_nodes).to(device)
    A = A.float()
    I = torch.eye(data.num_nodes).to(device)
    D = torch.sum(A, dim=1)
    D = D.unsqueeze(1)

    model.load_state_dict(torch.load("load/save_vgae_pubmed.pth"))

    z = model.encode(x, edge_index)
    # z1 = ((A + I) @ z) / (D + 1)
    # z2 = ((A + I) @ z1) / (D + 1)
    z1 = (A @ z) / (D + 1) + z
    z2 = (A @ z1) / (D + 1) + z
    z = F.normalize(z)
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    one = torch.ones(data.num_nodes, data.num_nodes).to(device)

    Sim00 = torch.mm(z, z.t())
    Sim00 = (Sim00 + one) / 2

    Sim02 = torch.mm(z, z2.t())
    Sim02 = (Sim02 + one) / 2

    Sim12 = torch.mm(z1, z2.t())
    Sim12 = (Sim12 + one) / 2

    Sim22 = torch.mm(z2, z2.t())
    Sim22 = (Sim22 + one) / 2


# Sim = torch.clamp(Sim, min=0)
    torch.save(Sim00, "res/pubmed_vgae00.pt")
    torch.save(Sim22, "res/pubmed_vgae22.pt")
    torch.save(Sim12, "res/pubmed_vgae12.pt")
    torch.save(Sim02, "res/pubmed_vgae02.pt")



