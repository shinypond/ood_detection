# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 03:50:34 2021

@author: Jaemoo
"""
import torch

class basic_CNN(torch.nn.Module):

    def __init__(self, image_size, nc):
        super(basic_CNN, self).__init__()
        #self.keep_prob = 0.5
        # L1 ImgIn shape=(?, im, im, 1)
        #    Conv     -> (?, im, im, 32)
        #    Pool     -> (?, im/2, im/2, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(nc, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, im/2, im.2, 32)
        #    Conv      ->(?, im/2, im/2, 64)
        #    Pool      ->(?, im/4, im/4, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, im/4, im/4, 64)
        #    Conv      ->(?, im/4, im/4, 128)
        #    Pool      ->(?, im/8, im/8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(image_size//8 * image_size//8 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),)
            #torch.nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out