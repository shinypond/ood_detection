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
from basic_CNN import basic_CNN as CNN
import os
from datetime import datetime


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=1, help='input image channels')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
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
    """
    dataset_fmnist_train = dset.FashionMNIST(root=opt.dataroot, train=True, download=True, transform=transforms.Compose([
                                transforms.Resize((opt.imageSize)),
                                transforms.ToTensor(),
                            ]))
    dataloader_fmnist = torch.utils.data.DataLoader(dataset_fmnist_train, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))
    """
    
    
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
        experiment = '../../../saved_models/CNN_cifar10/'
        dataset = dset.CIFAR10(
            root=opt.dataroot,
            download=False,
            train=True,
            transform=transform,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
        )
    elif traindist == '2':
        opt.nc = 1
        opt.niter = 200
        opt.train_dist = 'fmnist'
        experiment = '../../../saved_models/CNN_fmnist/'
        dataset = dset.FashionMNIST(
            root=opt.dataroot,
            download=False,
            train=True,
            transform=transform,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers)
        )
    else:
        raise ValueError('Oops! Please insert 1 or 2. Bye~')
        
    name = experiment.split('/')[-1] # This is 'CNN_cifar10' or 'CNN_fmnist'
    print(f'Dataloader for {name} is ready !')
    print(f'Please see the path "{experiment}" for the saved model !')
    
#############################################################
#############################################################
#############################################################

    ngpu = int(opt.ngpu)
    nc = int(opt.nc)
    
    cnn = CNN(opt.imageSize, nc).to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=opt.lr, weight_decay = 3e-5)
    cnn.train()
    loss_fn = nn.CrossEntropyLoss(reduction = 'mean')
    start = datetime.now()

    for epoch in range(opt.niter):
        Total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            b = x.size(0)
            y_pred = cnn(x)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            Total_loss += loss 
        print(f'Epoch: {epoch} Loss: {Total_loss}, Elapsed Time : {datetime.now() - start}')
        if epoch % 50 == 49:
            torch.save(cnn.state_dict(), experiment + f'cnn_augment_{opt.augment}_epoch_{epoch+1}.pth')
    torch.save(cnn.state_dict(), experiment + f'cnn_augment_{opt.augment}_epoch_{opt.niter}.pth')
