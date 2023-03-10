#!/usr/bin/env python
# coding=utf-8
'''
@Author: lzwgiter & JK211
@Email: float311@163.com & jerryren2884@gmail.com
@Date: 2022-07-1 11:09:49
@LastEditor: JK211
LastEditTime: 2022-08-01 11:10:11
@Discription: This .py is the gossip environment of NS3 inherit from gym.Env
@Environment: python 3.8
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttenModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, outpu_dim):
        super(AttenModel, self).__init__()
        self.fcv = nn.Linear(input_dim, hidden_dim)
        self.fck = nn.Linear(input_dim, hidden_dim)
        self.fcq = nn.Linear(input_dim, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, outpu_dim)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)

        att = F.softmax(torch.mul(torch.bmm(q, k), mask) - 9e15 * (1 - mask), dim=2)

        out = torch.bmm(att, v)
        out = F.relu(self.fcout(out))
        return out


class BranchingQNetwork(nn.Module):

    def __init__(self, obs, ac_dim, n_1, n_2):
        super().__init__()

        self.ac_dim = ac_dim
        self.n_1 = n_1
        self.n_2 = n_2

        self.model = AttenModel(obs, 128, 128)

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n_1), nn.Linear(128, n_2)])

    def forward(self, x, mask):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        out = self.model(x, mask).squeeze(0)
        value = self.value_head(out)
        adv_1 = self.adv_heads[0](out)
        adv_2 = self.adv_heads[1](out)

        q_val_1 = value + adv_1 - adv_1.mean(1, keepdim=True)
        q_val_2 = value + adv_2 - adv_2.mean(1, keepdim=True)

        return q_val_1, q_val_2
