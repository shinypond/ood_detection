import os, sys, argparse, random, copy
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

from data_loader import TRAIN_loader, TEST_loader
from train_GLOW.model_GLOW import Glow
import config

# fix a random seed
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) 
        return KL


def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
      
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = z - mu
        z_eps = z_eps.view(opt.repeat,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)
        
        weights = log_p_x_z+log_p_z-log_q_z_x
        
    return weights

def compute_NLL(weights):
    
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max()) 
        
    return NLL_loss

if __name__=="__main__":

    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Q. Dataset?')
    print('(1) CIFAR-10 (2) FMNIST')
    ans = input('-> ')
    if ans == '1':
        opt = config.GLOW_cifar10
        image_shape = (32, 32, 3)
        num_classes = 10
        split = True
        decay = '5e-05'
        epoch = 50
        end = '.pt'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = Glow(
            image_shape,
            opt.hidden_channels,
            opt.K,
            opt.L,
            opt.actnorm_scale,
            opt.flow_permutation,
            opt.flow_coupling,
            opt.LU_decomposed,
            num_classes,
            opt.learn_top,
            opt.y_condition,
            split,
        )
        model_path = f'{opt.modelroot}/GLOW_{opt.train_dist}/glow_{opt.train_dist}_decay_{decay}_epoch_{epoch}{end}'
        model.load_state_dict(torch.load(model_path)['model']) # there are two keys: 'model', 'optimizer'
        model.set_actnorm_init()
        model = model.to(device)
        model = model.eval()
        
        model_bg = Glow(
            image_shape,
            opt.hidden_channels,
            opt.K,
            opt.L,
            opt.actnorm_scale,
            opt.flow_permutation,
            opt.flow_coupling,
            opt.LU_decomposed,
            num_classes,
            opt.learn_top,
            opt.y_condition,
            split,
        )
        model_bg_path = f'{opt.modelroot}/GLOW_{opt.train_dist}/LRatio_bg/glow_ratio_0.2_decay_5e-05_epoch_50.pth'
        model_bg.load_state_dict(torch.load(model_bg_path)['model']) # there are two keys: 'model', 'optimizer'
        model_bg.set_actnorm_init()
        model_bg = model_bg.to(device)
        model_bg = model_bg.eval()
        
    elif ans == '2':
        opt = config.GLOW_fmnist
        image_shape = (32, 32, 1)
        num_classes = 10
        split = False
        decay = '0'
        epoch = 50
        end = '.pth'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = Glow(
            image_shape,
            opt.hidden_channels,
            opt.K,
            opt.L,
            opt.actnorm_scale,
            opt.flow_permutation,
            opt.flow_coupling,
            opt.LU_decomposed,
            num_classes,
            opt.learn_top,
            opt.y_condition,
            split,
        )
        model_path = f'{opt.modelroot}/GLOW_{opt.train_dist}/glow_{opt.train_dist}_decay_{decay}_epoch_{epoch}{end}'
        model.load_state_dict(torch.load(model_path))
        model.set_actnorm_init()
        model = model.to(device)
        model = model.eval()
        
        model_bg = Glow(
            image_shape,
            opt.hidden_channels,
            opt.K,
            opt.L,
            opt.actnorm_scale,
            opt.flow_permutation,
            opt.flow_coupling,
            opt.LU_decomposed,
            num_classes,
            opt.learn_top,
            opt.y_condition,
            split,
        )
        model_bg_path = f'{opt.modelroot}/GLOW_{opt.train_dist}/LRatio_bg/glow_ratio_0.3_decay_0.0_epoch_50.pth'
        model_bg.load_state_dict(torch.load(model_bg_path)['model']) # there are two keys: 'model', 'optimizer'
        model_bg.set_actnorm_init()
        model_bg = model_bg.to(device)
        model_bg = model_bg.eval()
        
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    assert opt.dataroot == '../../data', 'please go config.py and modify dataroot to "../../data"'
    
    print('Q. Test Num?')
    test_num = int(input('-> '))

    ######################################################################
    
    for ood in opt.ood_list:
        
        torch.manual_seed(2021)
        random.seed(2021)
        dataloader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=True,
        )

        NLL_test_ood = []
        NLL_test_ood_bg = []
        
        for i, x in enumerate(dataloader):
            try:
                x, _ = x
            except:
                pass
            with torch.no_grad():
                x = x.to(device)
                _, NLL_loss, _ = model(x, None)
                NLL_loss = NLL_loss.detach().cpu().numpy()[0]
                _, NLL_loss_bg, _ = model_bg(x, None)
                NLL_loss_bg = NLL_loss_bg.detach().cpu().numpy()[0]
                
                NLL_test_ood.append(NLL_loss.detach().cpu().numpy())
                NLL_test_ood_bg.append(NLL_loss_bg.detach().cpu().numpy())
                diff = -NLL_loss.item() + NLL_loss_bg.item()
                print(f'{ood} VAE: image {i} NLL {NLL_loss.item()}, NLL BG {NLL_loss_bg.item()}, diff: {diff}')
                
            if i >= test_num - 1:
                break

        NLL_test_ood = np.asarray(NLL_test_ood)
        NLL_test_ood_bg = np.asarray(NLL_test_ood_bg)
        metric_ood = -NLL_test_ood + NLL_test_ood_bg
        
        np.save(f'../npy/LRatio_GLOW/{opt.train_dist}_{ood}.npy', metric_ood)
        print(f'SAVED {opt.train_dist}_{ood} LRatio_GLOW npy !')
