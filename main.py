#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import torch
from utils.options import args_parser
from utils.load_dataset import load_dataset
from models.load_model import load_model
from models.train import Fed_train
import random
import numpy as np


if __name__ == '__main__':
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    # random seed
    def seed_torch(seed=1):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
    seed_torch(seed=args.seed)

    # logging
    LOG_FOUT = open('./log/log.txt', 'a')
    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)
    log_string("Poison_point: {}".format(args.point))

    # load dataset
    dataset_train, dataset_test, dict_users = load_dataset(args.dataset, args.dataset_dir, args.iid, args.num_clients)

    # load model
    img_size = dataset_train[0][0].shape
    net_glob = load_model(args.model, args.dataset, num_classes=args.num_classes, disable_dp=args.disable_dp)

    # training
    net_glob.train()
    net_glob = Fed_train(args, net_glob, dataset_train, dict_users, dataset_test, log_string )

