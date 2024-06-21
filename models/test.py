#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


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

