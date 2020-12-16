#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:58:25 2019

@author: YuxuanLong
"""

import numpy as np
import torch
from torch.autograd import Variable
import model
from loss import Loss


# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)


def np_to_var(x):
    """
    Convert numpy array to Torch variable.
    """
    x = torch.from_numpy(x)
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)


def var_to_np(x):
    """
    Convert Torch variable to numpy array.
    """
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def normalize(M):
    s = np.sum(M, axis = 1)
    s[s == 0] = 1
    return (M.T / s).T


def create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v,
                 side_hidden_dim, side_feature_u, side_feature_v,
                  use_side, use_GAT, out_dim, user_item_matrix_train, drop_out=0.0):
    """
    Choose one model from our implementations
    """
    side_feature_u = np_to_var(side_feature_u.astype(np.float32))
    side_feature_v = np_to_var(side_feature_v.astype(np.float32))
    
    for i in range(rate_num):
        all_M_u[i] = to_sparse(np_to_var(all_M_u[i].astype(np.float32)))
        all_M_v[i] = to_sparse(np_to_var(all_M_v[i].astype(np.float32)))   
    
    feature_u = to_sparse(np_to_var(feature_u.astype(np.float32)))
    feature_v = to_sparse(np_to_var(feature_v.astype(np.float32)))

    net = model.GCMC(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v, use_side, use_GAT, out_dim, user_item_matrix_train, drop_out)

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net


def epsilon_similarity_graph(X: np.ndarray, sigma=None, epsilon=0):
    """ X (n x d): coordinates of the n data points in R^d, d: feature dimension
        sigma (float): width of the kernel
        epsilon (float): threshold
        Return:
        adjacency (n x n ndarray): adjacency matrix of the graph.
    """
    # X: 1.side_feature_u[943,21] 2.side_feature_v[1682,19]
    W = np.array([np.sum((X[i] - X)**2, axis=1) for i in range(X.shape[0])])
    typical_dist = np.mean(np.sqrt(W))
    # print(np.mean(W))
    c = 0.35
    if sigma == None:
        sigma = typical_dist * c
    
    mask = W >= epsilon
    
    adjacency = np.exp(- W / 2.0 / (sigma ** 2))
    adjacency[mask] = 0.0
    adjacency -= np.diag(np.diag(adjacency))
    return adjacency

def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    d = np.sum(adjacency, axis=1)
    d_sqrt = np.sqrt(d)  
    if normalize:
        L = np.eye(adjacency.shape[0]) - (adjacency.T / d_sqrt).T / d_sqrt
    else:
        L = np.diag(d) - adjacency
    return L


def loss(all_M, mask, user_item_matrix, laplacian_loss_weight):
    all_M = np_to_var(all_M.astype(np.float32))
    mask = np_to_var(mask.astype(np.float32))
    user_item_matrix = np_to_var(user_item_matrix.astype(np.float32))
    
    return Loss(all_M, mask, user_item_matrix, laplacian_loss_weight)