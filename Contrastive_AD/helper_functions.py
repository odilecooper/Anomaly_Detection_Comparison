import torch
import numpy as np
import random
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def scores_calc_internal(query, positive, no_negatives, tau):
    pos_multiplication = (query * positive).sum(dim=2).unsqueeze(2).to(device)
    if no_negatives <= query.shape[1]:
        negative_index = random.sample(range(0, query.shape[1]), no_negatives)
    else:
        negative_index = random.sample(range(0, query.shape[1]), query.shape[1])
    neg_multiplication = torch.matmul(query, positive.permute(0, 2, 1)[:, :,negative_index])
    # Removal of the diagonals
    identity_matrix = torch.eye(np.shape(query)[1]).unsqueeze(0).repeat(np.shape(query)[0], 1,
                                                                        1)[:, :, negative_index].to(device)
    neg_multiplication.masked_fill_(identity_matrix == 1, -float('inf'))  # exp of -inf=0
    logits = torch.cat((pos_multiplication, neg_multiplication), dim=2).to(device)
    logits = logits / tau
    return (logits)


def take_per_row_complement(A, indx, num_elem=3):
    all_indx = indx[:,None] + np.arange(num_elem)

    all_indx_complement=[]
    for row in all_indx:
        complement=a_minus_b(np.arange(A.shape[2]),row)
        all_indx_complement.append(complement)
    all_indx_complement=np.array(all_indx_complement)
    return (A[:,np.arange(all_indx.shape[0])[:,None],all_indx],A[:,np.arange(all_indx.shape[0])[:,None],all_indx_complement])

def positive_matrice_builder(dataset, kernel_size):
    dataset = torch.squeeze(dataset, 2)
    if kernel_size != 1:
        indices = np.array((range(dataset.shape[1])))[:-kernel_size + 1]
    else:
        indices = np.array((range(dataset.shape[1])))
    dataset = torch.unsqueeze(dataset, 1)
    dataset = dataset.repeat(1, dataset.shape[2], 1)

    matrice,complement_matrice = take_per_row_complement(dataset, indices, num_elem=kernel_size)
    return (matrice,complement_matrice)


def take_per_row(A, indx, num_elem=2):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[:, np.arange(all_indx.shape[0])[:, None], all_indx]


def f1_calculator(classes, losses):
    # classes=classes.numpy()
    # losses=losses.numpy()
    df_version_classes = pd.DataFrame(data=classes)
    df_version_losses = pd.DataFrame(losses.cpu()).astype(np.float64)
    Na = df_version_classes[df_version_classes.iloc[:, 0] == 1].shape[0] # number of true anomalies
    anomaly_indices = df_version_losses.nlargest(Na, 0).index.values
    picked_anomalies = df_version_classes.iloc[anomaly_indices] # predicted anomalies
    true_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 1].shape[0]
    false_pos = picked_anomalies[picked_anomalies.iloc[:, 0] == 0].shape[0]
    f1 = true_pos / (true_pos + false_pos)
    return (f1)


def a_minus_b (a,b):
    sidx = b.argsort()
    idx = np.searchsorted(b, a, sorter=sidx)
    idx[idx == len(b)] = 0
    out = a[b[sidx[idx]] != a]
    return out