# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 03:49:23 2021

@author: Jaemoo
"""


import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
#from basic_CNN import basic_CNN as CNN
from CNN_models import CNN_cifar10, CNN_fmnist
from resnet import ResNet18
import os
from datetime import datetime

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias != None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias != None:
                nn.init.constant_(m.bias, 0)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')

    opt = parser.parse_args()
    opt.dataroot = '../../../data'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    
#############################################################
#############################################################
#############################################################

    # Ask train-dataset (1) CIFAR-10 (2) FMNIST 
    
    traindist = input('Q. Which dataset do you want to train the model for??\n(1) CIFAR-10 (2) FMNIST (Press 1 or 2)\n')
    augment = input('Q. Apply data augmentation?\n(1) None (2) hflip (Press 1 or 2)\n')
    
    if augment == '1':
        opt.augment = 'None'
        transform = transforms.Compose([
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ])
    elif augment == '2':
        opt.augment = 'hflip'
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ])
        # if you want to add other augmentations, then append them at this point!
    else:
        raise ValueError('Oops! Please insert 1 or 2. Bye~')
    
    if traindist == '1':
        opt.nc = 3
        opt.niter = 200
        opt.train_dist = 'cifar10'
        experiment = '../../../saved_models/CNN_cifar10'
        transform = transforms.Compose([
            transform,
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = dset.CIFAR10(
            root=opt.dataroot,
            download=False,
            train=True,
            transform=transform,
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            shuffle=True,
            num_workers=int(opt.workers),
        )
        #cnn = CNN_cifar10(opt.imageSize, opt.nc).to(device)
        
    elif traindist == '2':
        opt.nc = 1
        opt.niter = 200
        opt.train_dist = 'fmnist'
        experiment = '../../../saved_models/CNN_fmnist'
        transform = transforms.Compose([
            transform,
            transforms.Normalize((0.48,), (0.2,)),
        ])
        dataset = dset.FashionMNIST(
            root=opt.dataroot,
            download=False,
            train=True,
            transform=transform,
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers)
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            shuffle=True,
            num_workers=int(opt.workers)
        )
        #cnn = CNN_fmnist(opt.imageSize, opt.nc).to(device)
    else:
        raise ValueError('Oops! Please insert 1 or 2. Bye~')
        
    name = experiment.split('/')[-1] # This is 'CNN_cifar10' or 'CNN_fmnist'
    print(f'Dataloader for {name} is ready !')
    print(f'Please see the path "{experiment}" for the saved model !')
    
#############################################################
#############################################################
#############################################################

    cnn = ResNet18(opt.nc).to(device)
    cnn.apply(init_weights)
    optimizer = optim.SGD(cnn.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter)
    cnn.train()
    loss_fn = nn.CrossEntropyLoss()
    start = datetime.now()

    for epoch in range(opt.niter):
        # train
        Total_train_loss = 0
        train_correct = 0
        train_total = 0
        cnn.train()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = cnn(x)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            Total_train_loss += loss.item()
            _, predicted = y_pred.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
        # validation
        val_correct = 0
        cnn.eval()
        for i, (x, y) in enumerate(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            _, predicted = cnn(x).max(1)
            val_correct += predicted.eq(y).sum().item()
            if i == 1000:
                break
        # Print        
        print(f'Epoch: {epoch}, Train_Loss: {Total_train_loss:.2f}, Train_Acc : {100 * train_correct / train_total:.2f}%, Val_Acc : {100 * val_correct / 1000}%, Elapsed Time : {datetime.now() - start}')
        if epoch % 10 == 9:
            torch.save(cnn.state_dict(), experiment + f'/cnn_augment_{opt.augment}_epoch_{epoch+1}.pth')
            print(' < model checkpoint saved! > ')
    torch.save(cnn.state_dict(), experiment + f'/cnn_augment_{opt.augment}_epoch_{opt.niter}.pth')

    
    
    
    