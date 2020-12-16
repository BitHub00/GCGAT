#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:26:33 2019

@author: YuxuanLong
"""

import numpy as np
import torch
import utils
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import sys
import os 
import json
from dataset import prepare

parser = argparse.ArgumentParser(description='传入参数.')
parser.add_argument('--rate_num', type=int, default=5)
parser.add_argument('--use_side_feature', default=1, type=int, help='是否使用特征信息')
parser.add_argument('--lr', type=float, default=1e-2, help='学习率')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='衰减因子')
parser.add_argument('--use_GAT', type=int, default=1, help='是否引入attention机制')
parser.add_argument('--num_epochs', type=int, default=100, help='迭代次数')
parser.add_argument('--hidden_dim', type=int, default=5, help='隐层大小')
parser.add_argument('--side_hidden_dim', type=int, default=5, help='特征的隐层大小')
parser.add_argument('--out_dim', type=int, default=5, help='输出层大小')
parser.add_argument('--drop_out', type=float, default=0.0, help='dropout概率')
parser.add_argument('--split_ratio', type=float, default=0.8, help='训练集分割比例')
parser.add_argument('--save_steps', type=int, default=100, help='存储模型参数的频率')
parser.add_argument('--saved_model_folder', default='./weights', help='存储模型')
parser.add_argument('--laplacian_loss_weight', default=0.05, type=float, help='损失函数权重')
parser.add_argument('--dataset_path', default='../ml-100k', help='数据集路径')
parser.add_argument('--save_processed_data_path', default='../data', help='预处理数据存储路径')

args = parser.parse_args()


# 如果有GPU在GPU上运行
RUN_ON_GPU = torch.cuda.is_available()

# 设定随机数种子
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)

def validate(score, rate_num, user_item_matrix_test):
    sm = nn.Softmax(dim=0)
    score = sm(score)
    score_list = torch.split(score, rate_num)
    pred = 0
    for i in range(rate_num):
        pred += (i + 1) * score_list[0][i]

    pred = utils.var_to_np(pred)

    test_mask = user_item_matrix_test > 0

    square_err = (pred * test_mask - user_item_matrix_test) ** 2
    mse = square_err.sum() / test_mask.sum()
    test_rmse = np.sqrt(mse)
    
    return test_rmse


def main(args):
    
    # 获取参数
    rate_num = args.rate_num
    use_side_feature = args.use_side_feature  # using side feature or not
    use_GAT = args.use_GAT
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim
    side_hidden_dim = args.side_hidden_dim
    out_dim = args.out_dim
    drop_out = args.drop_out
    split_ratio = args.split_ratio
    save_steps = args.save_steps
    saved_model_folder = args.saved_model_folder
    laplacian_loss_weight = args.laplacian_loss_weight

    post_fix = '/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # 数据预处理
    feature_u, feature_v, feature_dim, all_M_u, all_M_v, side_feature_u, side_feature_v, all_M, mask,\
    user_item_matrix_train, user_item_matrix_test, laplacian_u, laplacian_v = prepare(args)

    if not os.path.exists(saved_model_folder):
        os.makedirs(saved_model_folder)  
    weights_name = saved_model_folder + post_fix + '_weights'

    net = utils.create_models(feature_u, feature_v, feature_dim, hidden_dim, rate_num, all_M_u, all_M_v,
                              side_hidden_dim, side_feature_u, side_feature_v,
                              use_side_feature, use_GAT, out_dim, user_item_matrix_train, drop_out)
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    Loss = utils.loss(all_M, mask, user_item_matrix_train, laplacian_loss_weight)
    iter_bar = tqdm(range(num_epochs), desc='Iter (loss=X.XXX)')

    for epoch in iter_bar:

        optimizer.zero_grad()

        score = net.forward()

        loss = Loss.loss(score)

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            rmse = Loss.rmse(score)
            
            val_rmse = validate(score, rate_num, user_item_matrix_test)
            iter_bar.set_description('Iter (loss=%5.3f, rmse=%5.3f, val_rmse=%5.5f)'%(loss.item(), rmse.item(), val_rmse.item()))


        if epoch % save_steps == 0:
            torch.save(net.state_dict(), weights_name)

    rmse = Loss.rmse(score)
    print('Final training RMSE: ', rmse.data.item())        
    torch.save(net.state_dict(), weights_name)
    
    sm = nn.Softmax(dim = 0)
    score = sm(score)
    score_list = torch.split(score, rate_num)
    pred = 0
    for i in range(rate_num):
        pred += (i + 1) * score_list[0][i]

    pred = utils.var_to_np(pred)

    test_mask = user_item_matrix_test > 0

    square_err = (pred * test_mask - user_item_matrix_test) ** 2
    mse = square_err.sum() / test_mask.sum()
    test_rmse = np.sqrt(mse)
    print('Test RMSE: ', test_rmse)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)