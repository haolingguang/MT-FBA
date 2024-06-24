#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=500, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')

    # poison arguments
    parser.add_argument( "--disable_BA", type=bool, default=False, help="Disable backdoor attack just clean training")
    parser.add_argument('--point', type=int, default=1, help="numbers of poisoned clients")
    parser.add_argument('--num_poison', type=int, default=30, help="numbers of poisoned clients")
    parser.add_argument('--trigger_round', type=int, default=40, help="")
    parser.add_argument('--trigger_train_epoch', type=int, default=10, help="")
    
    # defense arguments
    parser.add_argument( "--disable_dp", type=bool, default=True, help="Disable defense just clean training")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--dataset_dir', type=str, default='./data/cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_true',default=True,  help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--image_size', type=list, default=[3,32,32], help="size of imges")
    parser.add_argument('--gpu', type=str, default='7', help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args
