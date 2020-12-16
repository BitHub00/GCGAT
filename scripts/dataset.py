import numpy as np
import torch
import utils
from utils import epsilon_similarity_graph, compute_laplacian
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
import pandas as pd 
import numpy as np 
import random
import matplotlib.pyplot as plt
import os


def preprocess(dataset_path, save_path):
    """

    Args:
        dataset_path: '../ml-100k'
        save_path: '../data'

    Returns:

    """
    if os.path.exists(save_path+"/user_item_matrix.npy"):
        user_item_mat = np.load(save_path+"/user_item_matrix.npy", allow_pickle=True)
    else:
        data = pd.read_csv(dataset_path + '/u.data', sep='\t', names=["user_id", "item_id", "rating", "timestamp"])

        # The full u data set, 100000 ratings by 943 users on 1682 items
        user_item_mat = np.zeros((len(set(data["user_id"])), len(set(data["item_id"]))))

        for i in range(len(set(data["user_id"]))):
            for item_id in data[data['user_id'] == i+1]['item_id']:
                user_item_mat[i, item_id-1] = data[(data['user_id'] == i+1) & (data['item_id'] == item_id)]['rating']

    if os.path.exists(save_path+"/item_data_np.npy"):
        item_data_np = np.load(save_path+"/item_data_np.npy", allow_pickle=True)
    else:
        # genre_data[19,2]
        genre_data = pd.read_csv(dataset_path+'/u.genre', sep='|', names=["movie_type", "type_id"])

        # occupation_data[21,2]
        occupation_data = pd.read_csv(dataset_path+'/u.occupation', sep='|', names=["occupation"])
        occupation_data = occupation_data.reset_index().rename(columns={'index': 'occupation_id'})

        column_names = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", \
                      "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", \
                      "Sci-Fi", "Thriller", "War", "Western"]
        item_data_raw = pd.read_csv(dataset_path+'/u.item', sep='|', names=column_names, encoding = "ISO-8859-1")

        movie_headers = ['movie_id', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        # 特征筛选，去掉"movie_title"等
        item_data = item_data_raw[movie_headers]

        item_data = item_data.sort_values('movie_id', inplace=False)

        final_movie_headers = ['unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        # 排序后去掉特征"movie_id"
        item_data = item_data[final_movie_headers]

        item_data_np = item_data.to_numpy()

    if os.path.exists(save_path+"/user_data_np.npy"):
        user_data_np = np.load(save_path+"/user_data_np.npy", allow_pickle=True)
    else:
        
        column_names = ["user_id", "age", "gender", "occupation", "zip_code"]
        user_data = pd.read_csv(dataset_path+'/u.user', sep='|', names=column_names)
        ## convert all string information to numerical information ##
        # gender: M->0, F->1
        # occupation_type -> occupation_id

        user_data['gender'].replace('M', 0, inplace=True)
        user_data['gender'].replace('F', 1, inplace=True)

        # (0, 'technician')->(0, 19)
        new_user_data = user_data.merge(occupation_data, on='occupation').drop(columns=["occupation"])

        final_user_data = new_user_data[['user_id', 'age', 'gender', 'occupation_id']]

        final_user_data = final_user_data.sort_values("user_id", inplace=False)

        age_max = np.max(final_user_data[['age']])

        # 对age进行归一化
        final_user_data[['age']] = final_user_data[['age']]/age_max

        # 排序后去掉特征"user_id"
        user_data_np = final_user_data[['age', 'gender', 'occupation_id']]

        num_genres = len(set(user_data_np["occupation_id"]))

        user_data_np = user_data_np.to_numpy()

        # user_data_np.shape: [943,3], 3->['age', 'gender', 'occupation_id'], num_genres: 21
        data = np.zeros((user_data_np.shape[0], num_genres))

        # 只复制age与gender这两列，occupation_id用one_hot编码表示
        data[:,:2] = user_data_np[:,:2]

        for i in range(data.shape[0]):
            data[i][int(user_data_np[i][2])]=1

        user_data_np = data

    np.save(save_path+"/user_data_np.npy", user_data_np)
    np.save(save_path+"/item_data_np.npy", item_data_np)
    np.save(save_path+"/user_item_matrix.npy", user_item_mat)

    # user_item_mat[943, 1682]
    # item_data_np[1682, 19], 19->电影类别
    # user_data_np[932, 21], 0、1->age, gender, 2~21->occupation的one_hot编码
    return user_item_mat, item_data_np, user_data_np

def prepare(args):
    
    dataset_path = args.dataset_path  # '../ml-100k'
    save_path = args.save_processed_data_path  # './weights'
    
    user_item_matrix, raw_side_feature_v, raw_side_feature_u = preprocess(dataset_path, save_path)
    # user_item_matrix[943, 1682], raw_side_feature_u[943, 21], raw_side_feature_u[1682,19]

    num_user, num_item = user_item_matrix.shape
    mask = user_item_matrix > 0
    mask_new = mask + np.random.uniform(0, 1, (num_user, num_item))
    train_mask = (mask_new <= (1 + args.split_ratio)) & mask
    test_mask = (mask_new > (1 + args.split_ratio)) & mask
    user_item_matrix_train = user_item_matrix + 0
    user_item_matrix_train[test_mask] = 0
    user_item_matrix_test = user_item_matrix + 0
    user_item_matrix_test[train_mask] = 0

    all_M_u = []
    all_M_v = []
    all_M = []
    for i in range(args.rate_num):  # rate_num==5, 5分制
        M_r = user_item_matrix_train == (i + 1)  # 用户u给物品v评分为r
        all_M_u.append(utils.normalize(M_r))  # 评分
        all_M_v.append(utils.normalize(M_r.T))  # 被评分
        all_M.append(M_r)
    # [[943,1682],[943,1682],[943,1682],[943,1682],[943,1682]] -> ndarray(5, 943, 1682), 5分制
    all_M = np.array(all_M)
    mask = user_item_matrix_train > 0   


    ## 读取side_information并处理

    side_feature_u = raw_side_feature_u
    side_feature_v = raw_side_feature_v

    adjacency_u = epsilon_similarity_graph(side_feature_u, epsilon=1.1)
    laplacian_u = compute_laplacian(adjacency_u, True)
    adjacency_v = epsilon_similarity_graph(side_feature_v, epsilon=2.1)
    laplacian_v = compute_laplacian(adjacency_v, True)

    laplacian_u = utils.np_to_var(laplacian_u)
    laplacian_v = utils.np_to_var(laplacian_v)


    ## 产生输入特征
    # 对应论文中的(Xu;Xv) = I
    feature_dim = num_user + num_item
    I = np.eye(num_user + num_item)
    feature_u = I[0:num_user, :]
    feature_v = I[num_user:, :]
    
    return feature_u, feature_v, feature_dim, all_M_u, all_M_v, side_feature_u, side_feature_v, all_M, mask, user_item_matrix_train, user_item_matrix_test, laplacian_u, laplacian_v
