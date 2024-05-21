import argparse

import torch
import yaml
import torch.nn.functional as F
import torch.nn as nn

from time import perf_counter as t
from model import Model
from yaml import SafeLoader
from dataset import load_dataset
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from utils import *
from eval import label_classification
from pre_train.vgae import incomplete_negative


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    loss = model(edge_index_1, edge_index_2, x_1, x_2)
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Photo')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config/config40.yaml')
    parser.add_argument('--lambda0', type=float, default='1.0')
    parser.add_argument('--lambda2', type=float, default='0.0')

    # parser.add_argument('--path', type=str, default='dataset/cora_sum_vgae.pt')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    print(args)
    print(config)

    set_seed(config['seed'])


    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']


    lambda0 = args.lambda0
    lambda2 = args.lambda2

    dataset = load_dataset(args.dataset)
    data = dataset[0]
    A = edge_index_to_adjacency_matrix(data.edge_index, data.num_nodes)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    A = A.to(device)
    DS = incomplete_negative(data)
    # DS = DS.to(device)


    model = Model(dataset.num_features, num_hidden, num_proj_hidden, num_layers, activation, tau, DS, lambda0, lambda2)
    model = model.to(device)

    print(f"# params: {count_parameters(model)}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    best_micro = 0.0
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

        """Evaluation Embeddings  """
        if epoch > num_epochs - 100:
            embeds = model.get_embedding(data.x, data.edge_index)
            res = label_classification(embeds, data.y, 0.2)
            if res['acc']['mean'] > best_acc:
                best_acc = res['acc']['mean']
                best_epoch = epoch
                torch.save(embeds, "citeseer_muxgcl.pt")
    #
    print("=== Final ===")
    print(f'Epoch={best_epoch}, best_acc={best_acc * 100:.2f}')

