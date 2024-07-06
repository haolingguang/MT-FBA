#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import torch
from utils.backdoor_image_dataset import Dataset
from models.load_model import load_model
from torch.utils.data import DataLoader

import random
import numpy as np
from Defense.Bit_Red import bit_depth_reduce
from Defense.FD import fd_image
from Defense.NRP import nrp
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--defense", type=str, default="NRP", help="defense type")
    parser.add_argument('--gpu', type=str, default='7', help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    args = parser.parse_args()    
    return args

args = args_parser()





if __name__ == '__main__':
    # parse args
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # torch.manual_seed(args.seed)
    
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
    
    # 记录日志
    LOG_FOUT = open('./log/log_backdoor.txt', 'a')
    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)
    
    # 加载模型
    net_glob = load_model(args.model, args.dataset, num_classes=args.num_classes)
    # MT-FBA
    net_glob.load_state_dict(torch.load('./checkpoints/resnet18.pth'))
    
    # DBA
    # net_glob.load_state_dict(torch.load('../CIFAR10_Target_poison_DBA/checkpoints/1/resnet18.pth'))

    net_glob.eval()

    if args.defense == "NRP":
        # load NRP model parameters
        nrp_ = nrp('./Defense/NRP_pretrained_purifiers/NRP.pth')
    

    # 加载数据集

    for idx in range(10):
        # backdoor image dir
        dataset_dir = "./save/backdoor_image/"+str(idx)    

        dataset_backdoor = Dataset(dataset_dir)
        dataloader_backdoor = DataLoader(dataset_backdoor, batch_size=args.local_bs, shuffle=False)
        attack_success = 0
        for _, (input, backdoor_label,filename) in enumerate(dataloader_backdoor):  

            input = input.cuda()
            if args.defense == "Bit_Red":
                denoise_images, _ = bit_depth_reduce(input, -1, 1, 4, 200)

            elif args.defense == "FD":
                denoise_images = input.cpu()
                for i in range(denoise_images.shape[0]):        
                    denoise_images[i] = fd_image(denoise_images[i])
                denoise_images = denoise_images.cuda()

            elif args.defense == "NRP":
                denoise_images = nrp_(input)

            backdoor_label = backdoor_label.cuda()
            _, prediction = torch.max(net_glob(denoise_images),1)
            attack_success +=  torch.sum(prediction == backdoor_label)


        attack_succ_rate = attack_success/len(dataset_backdoor)*100
        log_string('label: %d  |  success_rate: %.3f'%(idx, attack_succ_rate))




