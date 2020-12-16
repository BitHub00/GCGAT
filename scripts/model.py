#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:02:15 2019

@author: YuxuanLong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import numpy as np
import utils
from loss import Loss

def sparse_drop(feature, drop_out):
    tem = torch.rand((feature._nnz()))
    feature._values()[tem < drop_out] = 0
    return feature

class GCMC(nn.Module):
    def __init__(self, feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v, 
                 side_hidden_dim, side_feature_u, side_feature_v,
                 use_side, use_GAT, out_dim, user_item_matrix_train, drop_out=0.0):
        super(GCMC, self).__init__()
        
        self.drop_out = drop_out
        
        side_feature_u_dim = side_feature_u.shape[1]
        side_feature_v_dim = side_feature_v.shape[1]
        self.use_side = use_side  # using side feature or not
        self.use_GAT = use_GAT  # using GAT or not

        self.feature_u = feature_u
        self.feature_v = feature_v
        self.rate_num = rate_num
        
        self.num_user = feature_u.shape[0]
        self.num_item = feature_v.shape[1]
        
        self.side_feature_u = side_feature_u
        self.side_feature_v = side_feature_v
        
        self.W = nn.Parameter(torch.randn(rate_num, feature_dim, hidden_dim))  # 对应公式(4)与(5)之间的矩阵Wr, r对应一个rating
        nn.init.kaiming_normal_(self.W, mode='fan_out', nonlinearity='relu')

        if use_GAT:
            # 引入GAT思想
            self.W_att_u = nn.Parameter(torch.empty(size=(side_feature_u_dim, hidden_dim)))
            nn.init.xavier_uniform_(self.W_att_u.data, gain=1.414)
            self.W_att_v = nn.Parameter(torch.empty(size=(side_feature_v_dim, hidden_dim)))
            nn.init.xavier_uniform_(self.W_att_v.data, gain=1.414)
            self.a = nn.Parameter(torch.empty(size=(2 * hidden_dim, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
            self.leakyrelu = nn.LeakyReLU(0.2)
            # adj_u[5, 943, 943]
            self.adj_u = self._prepare_adjacency_matrix(user_item_matrix_train, user=True)
            # adj_v[5, 1682, 1682]
            self.adj_v = self._prepare_adjacency_matrix(user_item_matrix_train, user=False)
            self.hidden_dim = hidden_dim
        
        self.all_M_u = all_M_u
        self.all_M_v = all_M_v
        
        self.reLU = nn.ReLU()
        
        if use_side:
            self.linear_layer_side_u = nn.Sequential(*[nn.Linear(side_feature_u_dim, side_hidden_dim, bias=True),
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
            self.linear_layer_side_v = nn.Sequential(*[nn.Linear(side_feature_v_dim, side_hidden_dim, bias=True),
                                                       nn.BatchNorm1d(side_hidden_dim), nn.ReLU()])
    
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias=True),
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2 + side_hidden_dim, out_dim, bias=True),
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])    
        else:
            
            self.linear_cat_u = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias=True),
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])
            self.linear_cat_v = nn.Sequential(*[nn.Linear(rate_num * hidden_dim * 2, out_dim, bias=True),
                                                nn.BatchNorm1d(out_dim), nn.ReLU()])

        self.Q = nn.Parameter(torch.randn(rate_num, out_dim, out_dim))
        nn.init.orthogonal_(self.Q)
        
        
    def forward(self):
        
        feature_u_drop = sparse_drop(self.feature_u, self.drop_out) / (1.0 - self.drop_out)
        feature_v_drop = sparse_drop(self.feature_v, self.drop_out) / (1.0 - self.drop_out)

        hidden_feature_u = []
        hidden_feature_v = []
        
        W_list = torch.split(self.W, self.rate_num)
        if self.use_GAT:
            adj_u_list = torch.split(self.adj_u, self.rate_num)
            adj_v_list = torch.split(self.adj_v, self.rate_num)
        W_flat = []
        for i in range(self.rate_num):
            Wr = W_list[0][i]
            if self.use_GAT:
                adj_u = adj_u_list[0][i]
                adj_v = adj_v_list[0][i]
                att_u = self._calculate_attention(self.side_feature_u, adj_u, self.W_att_u)
                att_v = self._calculate_attention(self.side_feature_v, adj_v, self.W_att_v)
            M_u = self.all_M_u[i]
            M_v = self.all_M_v[i]
            hidden_u = sp.mm(feature_v_drop, Wr)
            if self.use_GAT:
                hidden_u = torch.matmul(att_v, hidden_u)
            hidden_u = self.reLU(sp.mm(M_u, hidden_u))
            
            hidden_v = sp.mm(feature_u_drop, Wr)
            if self.use_GAT:
                hidden_v = torch.matmul(att_u, hidden_v)
            hidden_v = self.reLU(sp.mm(M_v, hidden_v))

            
            hidden_feature_u.append(hidden_u)
            hidden_feature_v.append(hidden_v)
            
            W_flat.append(Wr)
            
        hidden_feature_u = torch.cat(hidden_feature_u, dim=1)
        hidden_feature_v = torch.cat(hidden_feature_v, dim=1)
        W_flat = torch.cat(W_flat, dim=1)

        cat_u = torch.cat((hidden_feature_u, torch.mm(self.feature_u, W_flat)), dim=1)
        cat_v = torch.cat((hidden_feature_v, torch.mm(self.feature_v, W_flat)), dim=1)
        
        if self.use_side:
            side_hidden_feature_u = self.linear_layer_side_u(self.side_feature_u)
            side_hidden_feature_v = self.linear_layer_side_v(self.side_feature_v)
            
            cat_u = torch.cat((cat_u, side_hidden_feature_u), dim=1)
            cat_v = torch.cat((cat_v, side_hidden_feature_v), dim=1)

        embed_u = self.linear_cat_u(cat_u)
        embed_v = self.linear_cat_v(cat_v)
        
        score = []
        Q_list = torch.split(self.Q, self.rate_num)
        for i in range(self.rate_num):
            Qr = Q_list[0][i]
            
            tem = torch.mm(torch.mm(embed_u, Qr), torch.t(embed_v))  # 对应公式(12)
            
            score.append(tem)
        return torch.stack(score)

    def _calculate_attention(self, h, adj, W):
        Wh = torch.mm(h, W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        #  torch.matmul(a_input, self.a)->torch.size([2708,2708,1])
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        #  zero_vec = -9e15 * torch.ones_like(e)
        #  torch.where(condition, x, y)->if condition:x else:y
        #  原来adj只是邻接矩阵，其中元素的值表示顶点i与j之间是否有边相连，
        #  现在attention中元素的值表示顶点j的特征对顶点i的重要程度，对应原文公式(3)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=1)
        return attention

    def _prepare_adjacency_matrix(self, user_item_matrix_train, user=True):
        if user:
            user_item_matrix_train = user_item_matrix_train.T
        adj = torch.empty(size=(self.rate_num, user_item_matrix_train.shape[1], user_item_matrix_train.shape[1]))
        for i in range(self.rate_num):
            m = torch.empty(size=(user_item_matrix_train.shape[1], user_item_matrix_train.shape[1]))
            rate = [i + 1]
            for j in range(user_item_matrix_train.shape[0]):
                interaction = user_item_matrix_train[j]
                u = np.where(np.isin(interaction, rate))[0]
                for k in range(len(u)):
                    for l in range(len(u)):
                        if k == l:  # 邻接矩阵对角线是否为1
                            continue
                        m[u[k]][u[l]] = 1
            adj[i, :, :] = m
        return adj

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        # >>> x = torch.tensor([1, 2, 3])
        # >>> x.repeat_interleave(2)
        # tensor([1, 1, 2, 2, 3, 3])
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        #  all_combinations_matrix->torch.Size([7333264, 16])
        #  Wh_repeated_in_chunks = Wh_repeated_alternating->torch.Size([7333264, 8])
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        #  view(N, N, 2 * self.out_features)->torch.Size([2708, 2708, 16])
        return all_combinations_matrix.view(N, N, 2 * self.hidden_dim)
    

        


