#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn as nn
import sys
from models.DP import Differencial_Privacy

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # 
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dp = Differencial_Privacy()

    def train(self, net, rounds):

        net.train()
        
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        
        # differential privacy 
        if not self.args.disable_dp:
             net, optimizer, self.ldr_train = self.dp.init_model(dataloader=self.ldr_train, optimizer=optimizer, net=net)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.cuda(), labels.cuda()
                
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        if not self.args.disable_dp:
            epsilon = self.dp.DP_epsilon()
            print(
                        f"\tTrain Epoch: {rounds} \t"
                        f"Loss: {(sum(epoch_loss)/len(epoch_loss)):.2f} "
                        f"(ε = {epsilon:.2f}, δ = {self.dp.delta})"
                    )

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_noise(self, net_glob, noise):

        net_glob.eval()
        
        # define bound
        l_inf_r = 16/255
        batch_opt = []
        batch_pert = []
        

        for i in range(len(noise)):
            batch_pert.append(torch.autograd.Variable(noise[i], requires_grad=True))
        for i in range(len(noise)):
            batch_opt.append(torch.optim.Adam(params=[batch_pert[i]], lr=0.01))  
             

        for minmin in range(self.args.trigger_train_epoch):
            loss_list = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.cuda(), labels.cuda()
                
                # zero gradient
                for i in range(len(batch_opt)):
                    batch_opt[i].zero_grad()
                
                # bound trigger
                clamp_batch_pert=[[] for i in range(len(batch_pert))]
                for i in range(len(batch_pert)):
                    clamp_batch_pert[i] = torch.clamp(batch_pert[i],-l_inf_r*2,l_inf_r*2)
                new_images = torch.clamp(apply_noise_patch(clamp_batch_pert, copy.deepcopy(images), labels),-1,1)
                

                per_logits = net_glob(new_images)
                loss = self.loss_func(per_logits, labels)
                loss.backward()
                loss_list.append(loss.item())
                for i in range(len(batch_opt)):
                    batch_opt[i].step()


        for i in range(len(batch_pert)):
            batch_pert[i] = torch.clamp(batch_pert[i].detach(),-l_inf_r*2,l_inf_r*2)
        return batch_pert
            
    def train_backoodr_model(self, net, net_glob, noise, rounds):

        net.train()
        net_glob.eval()
        

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # differential privacy 
        if self.args.disable_dp:
             net, optimizer, self.ldr_train = self.dp.init_model(dataloader=self.ldr_train, optimizer=optimizer, net=net)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.cuda(), labels.cuda()


                images_adv = torch.clamp(apply_noise_patch(noise,copy.deepcopy(images),labels),-1,1)

                log_adv = net(images_adv)
                loss_adv = self.loss_func(log_adv, labels)
                
                log_clean= net(images)
                loss_clean = self.loss_func(log_clean, labels)
                loss = (loss_clean + loss_adv)/2
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def apply_noise_patch(noise,images,labels):
    '''
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    '''

    for i in range(images.shape[0]):
        # noise_now = noise.clone()
        images[i] = images[i] + noise[labels[i]]
    return images
