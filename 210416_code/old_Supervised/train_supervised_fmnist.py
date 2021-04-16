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
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if opt.experiment is None:
        opt.experiment = './models/fmnist'
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

   # dataset = dset.CIFAR10(root=opt.dataroot, download=True,train = True,
    #                       transform=transforms.Compose([
     #                           transforms.Resize((opt.imageSize)),
      #                         transforms.ToTensor(),
       #                    ]))
   # dataloader_cifar = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
    #                                        shuffle=True, num_workers=int(opt.workers))
    
    dataset_fmnist_train = dset.FashionMNIST(root=opt.dataroot, train=True, download=True, transform=transforms.Compose([
                                transforms.Resize((opt.imageSize)),
                                transforms.ToTensor(),
                            ]))
    dataloader_fmnist = torch.utils.data.DataLoader(dataset_fmnist_train, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))
    
    
    ngpu = int(opt.ngpu)
    nc = int(opt.nc)
    
    cnn = CNN(opt.imageSize, nc).to(device)
    # setup optimizer
    
    optimizer = optim.Adam(cnn.parameters(), lr=opt.lr, weight_decay = 3e-5)

    cnn.train()

    loss_fn = nn.CrossEntropyLoss(reduction = 'mean')

    for epoch in range(opt.niter):
        Total_loss = 0
        for i, (x, y) in enumerate(dataloader_fmnist):
            x = x.to(device)
            y = y.to(device)
            b = x.size(0)
            y_pred = cnn(x)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            Total_loss += loss 
        print('epoch:{} recon:{}'.format(epoch,Total_loss))
    if not os.path.exists('./saved_models/{}'.format(opt.experiment)):
        os.makedirs('./saved_models/{}'.format(opt.experiment))
    torch.save(cnn.state_dict(), './saved_models/{}/cnn.pth'.format(opt.experiment))
