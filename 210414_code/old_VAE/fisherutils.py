# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:53:59 2021

@author: Jaemoo
"""

import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def KL_div(mu,logvar,reduction = 'avg'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1) 
        return KL

def Calculate_fisher(netE, netG, dataset, dicts, max_iter, num_samples, opt, device='cuda:0'):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                            shuffle=False, num_workers=0)
    optimizer1 = optim.SGD(netE.parameters(), lr=0,momentum=0)
    optimizer2 = optim.SGD(netG.parameters(), lr=0,momentum=0)
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    for i, (x, _) in enumerate(dataloader):

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x = x.repeat(num_samples,1,1,1).to(device)
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        [z,mu,logvar] = netE(x)
        kld = KL_div(mu,logvar)
        recon = netG(z)
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        recl = loss_fn(recon, target)
        recl = torch.sum(recl) / b
        loss =  recl + kld.mean()
        grads = []
        loss.backward(retain_graph=True)
        
        for param in dicts:
            grads.append(param.grad.view(-1)**2) 
        grads = torch.cat(grads)
        if i==0:
            Grads = grads
            
        else:
            Grads = (i*Grads + grads)/(i+1)
            if i>max_iter:
                break
            

    Grads = torch.sqrt(Grads)
    Grads = Grads*(Grads>1e-3)
    Grads[Grads==0] = 1e-3
    normalize_factor = 2*np.sqrt(len(Grads))
    return Grads, normalize_factor

def Calculate_score(netE, netG, dataset, dicts, Grads, normalize_factor, number, num_samples, opt, device='cuda:0'):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                            shuffle=False, num_workers=0)
    optimizer1 = optim.SGD(netE.parameters(), lr=0,momentum=0)
    optimizer2 = optim.SGD(netG.parameters(), lr=0,momentum=0)
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    score = []
    
    if len(dataset[0])==2:
        for i, (x, _) in enumerate(dataloader):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            x = x.repeat(num_samples,1,1,1).to(device)
            b = x.size(0)
            target = Variable(x.data.view(-1) * 255).long()
            gradient_val=0
            [z,mu,logvar] = netE(x)
            kld = KL_div(mu,logvar)
            recon = netG(z)
            recon = recon.contiguous()
            recon = recon.view(-1,256)
            recl = loss_fn(recon, target)
            loss =  torch.sum(recl)/b + kld.mean()
            loss.backward(retain_graph=True)
            grads = []
            for param in dicts:
                grads.append(param.grad.view(-1)) 
            grads = torch.cat(grads)
            gradient_val = torch.norm(grads/Grads)/normalize_factor
            score.append(gradient_val.detach().cpu())
            if i%number==number-1:
                break
    else:
        for i, x in enumerate(dataloader):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            x = x.repeat(num_samples,1,1,1).to(device)
            b = x.size(0)
            target = Variable(x.data.view(-1) * 255).long()
            gradient_val=0
            [z,mu,logvar] = netE(x)
            kld = KL_div(mu,logvar)
            recon = netG(z)
            recon = recon.contiguous()
            recon = recon.view(-1,256)
            recl = loss_fn(recon, target)
            loss =  torch.sum(recl)/b + kld.mean()
            loss.backward(retain_graph=True)
            grads = []
            for x in dicts:
                for param in x.parameters():
                    grads.append(param.grad.view(-1)) 
            grads = torch.cat(grads)
            gradient_val = torch.norm(grads/Grads)/normalize_factor
            score.append(gradient_val.detach().cpu())
            if i%number==number-1:
                break
    return score

def plot_hist(indist_Grads, outdist_Grads, bins1=100, bins2 =200):
    indist_Grads = torch.tensor(indist_Grads)
    plt.hist(indist_Grads, bins=bins1)

    outdist_Grads = torch.tensor(outdist_Grads)
    plt.hist(outdist_Grads, bins=bins2, alpha=0.5)
    plt.xlim(0,10)


def AUROC(indist_Grads, outdist_Grads):    
    combined = np.concatenate((indist_Grads, outdist_Grads))
    label_1 = np.ones(len(indist_Grads))
    label_2 = np.zeros(len(outdist_Grads))
    label = np.concatenate((label_1, label_2))
    fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
    #plot_roc_curve(fpr, tpr)
    rocauc = metrics.auc(fpr, tpr)
    print('AUC for Gradient Norm is: {}'.format(rocauc))
    plt.plot(fpr, tpr)
    plt.show()