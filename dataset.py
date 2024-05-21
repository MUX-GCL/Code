from torch_geometric.datasets import Planetoid, CitationFull, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
import torch

def load_dataset(name):
    if name == 'Cora':
        dataset = Planetoid(root='/data/Cora', name='Cora')
    elif name == 'CiteSeer':
        dataset = Planetoid(root='/data/Citeseer', name='Citeseer')
    elif name == 'PubMed':
        dataset = Planetoid(root='/data/Pubmed', name='Pubmed')
    elif name == 'Photo':
        dataset = Amazon(root='/data/amazon-photo', name='Photo')
    elif name == 'Computer':
        dataset = Amazon(root='/data/amazon-computer', name='computers')
    elif name == 'Arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/data/arxiv/')
    else:
        print("no such dataset!")


    return dataset

