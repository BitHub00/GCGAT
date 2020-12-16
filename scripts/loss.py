import torch
import torch.nn as nn
import torch.sparse as sp
import numpy as np
import utils

class Loss(nn.Module):
    def __init__(self, all_M, mask, user_item_matrix, laplacian_loss_weight):
            
        super(Loss, self).__init__()
            
        self.all_M = all_M
        self.mask = mask
        self.user_item_matrix = user_item_matrix
        
        self.rate_num = all_M.shape[0]
        self.num = float(mask.sum())
        
        self.logsm = nn.LogSoftmax(dim = 0)
        self.sm = nn.Softmax(dim = 0)
        self.laplacian_loss_weight = laplacian_loss_weight
        
    def cross_entropy(self, score):
        l = torch.sum(-self.all_M * self.logsm(score))
        return l / self.num
    
    def rmse(self, score):
        score_list = torch.split(self.sm(score), self.rate_num)
        total_score = 0
        for i in range(self.rate_num):
            total_score += (i + 1) * score_list[0][i]
        
        square_err = torch.pow(total_score * self.mask - self.user_item_matrix, 2)
        mse = torch.sum(square_err) / self.num
        return torch.sqrt(mse)
        
    def loss(self, score):
        return self.cross_entropy(score) + self.rmse(score)