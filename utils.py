import torch
import numpy as np
import random
import os
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

from eval import prob_to_one_hot


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)

    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    adjacency_matrix[edge_index[1], edge_index[0]] = 1

    return adjacency_matrix

def evaluate_clustering(emb, nb_class, true_y, repetition_cluster):
    embeddings = F.normalize(emb, dim=-1, p=2).detach().cpu().numpy()
    true_y = true_y.detach().cpu().numpy()
    estimator = KMeans(n_clusters = nb_class)

    NMI_list = []
    ARI_list = []

    for _ in range(repetition_cluster):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)

        nmi_score = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(true_y, y_pred)
        NMI_list.append(nmi_score)
        ARI_list.append(ari_score)

    return np.mean(NMI_list), np.std(NMI_list), np.mean(ARI_list), np.std(ARI_list)


def public_classification(embeddings, data):
    X = embeddings.detach().cpu().numpy()
    Y = data.y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().cpu()
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().cpu()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().cpu()
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = Y[train_idx]
    y_test = Y[test_idx]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy




