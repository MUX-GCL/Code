import torch
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.nn import Node2Vec
import torch.nn.functional as F
from eval import label_classification


def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)

    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    adjacency_matrix[edge_index[1], edge_index[0]] = 1

    return adjacency_matrix
#
# dataset = Planetoid(root='/data/Cora', name='Cora')
dataset = Amazon(root='/data/amazon-photo', name='Photo')
# dataset = Planetoid(root='/data/Pubmed', name='Pubmed')
# dataset = Planetoid(root='/data/Citeseer', name='Citeseer')
# dataset = Amazon(root='/data/amazon-computer', name='computers')
data = dataset[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# node2vec模型
model = Node2Vec(edge_index=data.edge_index,
                 embedding_dim=128,  # 节点维度嵌入长度
                 walk_length=20,  # 序列游走长度
                 context_size=10,  # 上下文大小
                 walks_per_node=10,  # 每个节点游走10个序列
                 p=1,
                 q=1,
                 num_negative_samples=1,
                 sparse=True  # 权重设置为稀疏矩阵
                ).to(device)
# 迭代器
# loader = model.loader(batch_size=128, shuffle=True)
# # 优化器
# optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

# model.load_state_dict(torch.load('save_model.pth'))
# z = model()  # 获取权重系数，也就是embedding向量表
#
# # z[data.train_mask] 获取训练集节点的embedding向量
# acc = model.test(z[data.train_mask], data.y[data.train_mask],
#                  z[data.test_mask], data.y[data.test_mask],
#                  max_iter=150)
# print(acc)
# 3.开始训练
# model.train()
#
# for epoch in range(1, 51):
#     total_loss = 0 # 每个epoch的总损失
#     print("epoch:", epoch)
#     for pos_rw, neg_rw in loader:
#         optimizer.zero_grad()
#         loss = model.loss(pos_rw.to(device), neg_rw.to(device)) # 计算损失
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

    # 使用逻辑回归任务进行测试生成的embedding效果
    # with torch.no_grad():
    #     model.eval() # 开启测试模式
    #     z = model() # 获取权重系数，也就是embedding向量表
    #
    #     label_classification(z, data.y, ratio=0.1)
    # 打印指标


A = edge_index_to_adjacency_matrix(data.edge_index, data.num_nodes).to(device)
A = A.float()
I = torch.eye(data.num_nodes).to(device)
D = torch.sum(A, dim=1)
D = D.unsqueeze(1)

model.load_state_dict(torch.load("load/save_model_photo.pth"))

z = model()
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
torch.save(Sim00, "res/photo_DW00.pt")
torch.save(Sim22, "res/photo_DW22.pt")
torch.save(Sim12, "res/photo_DW12.pt")
torch.save(Sim02, "res/photo_DW02.pt")

