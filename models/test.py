#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import copy

@torch.no_grad()
def test_img(net_g, datatest, args):

    net_g.eval()
    
    test_loss = 0
    correct = 0
    
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    
    
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        

        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        

        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()


    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    

    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


@torch.no_grad()
def test_backdoor(net_g, datatest, args, noise, target):

    net_g.eval()


    test_loss = 0
    correct = 0
    

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    

    for idx, (image, label) in enumerate(data_loader):


        image, label = image.cuda(), label.cuda()
        image += noise[target]
        image_clamp = torch.clamp(image,-1,1)

         # save backdoor image
        Save_Image(copy.deepcopy(image_clamp), idx, './save/backdoor_image/'+str(target))

        target_label = [target for i in range(len(label))]
        target_label = torch.tensor(target_label,dtype=torch.long).flatten().cuda()
        

        log_probs = net_g(image_clamp)
        test_loss += F.cross_entropy(log_probs, target_label, reduction='sum').item()

        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target_label.data.view_as(y_pred)).long().cpu().sum()


    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def Save_Image(Image, idx, output_dir):
    # 创建图像保存路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    for index in range(len(Image)):
        # 反归一化图像
        Image_norm = Image[index].div_(2).add(0.5)
        Image_path = os.path.join(output_dir, str(idx)+'_'+str(index)+'.png')
        save_image(Image_norm, Image_path)