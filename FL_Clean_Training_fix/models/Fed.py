#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(local_models, fracs):
    # 准备累加的模型
    weight_accumulator = {}
    for name, params in local_models[0].items():
        weight_accumulator[name] = torch.zeros_like(params, dtype=torch.float32)
    
    # 按数据量聚合
    fracs_norm = torch.tensor([frac/sum(fracs) for frac in fracs]).cuda()
    for i, data in enumerate(local_models):		
        for name, params in data.items():
            weight_accumulator[name] += params*fracs_norm[i]
    return weight_accumulator

# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg
