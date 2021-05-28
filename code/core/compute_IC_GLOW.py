import os, sys, argparse, random, copy
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
import cv2
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

sys.path.append(os.getcwd() + '/core')

from data_loader import TRAIN_loader, TEST_loader
from train_GLOW.model_GLOW import Glow
import config

# fix a random seed
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

def reparameterize(mu, logvar, device):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return eps.mul(std).add_(mu)

def KL_div(mu,logvar,reduction = 'none'):
    mu = mu.view(mu.size(0), mu.size(1))
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
        recon = recon.view(-1, 256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b, -1), 1)
        log_p_z = -torch.sum(z**2 / 2 + np.log(2 * np.pi) / 2,1)
        z_eps = z - mu
        z_eps = z_eps.view(opt.repeat, -1)
        log_q_z_x = -torch.sum(z_eps**2 / 2 + np.log(2 * np.pi) / 2 + logvar / 2, 1)
        weights = log_p_x_z + log_p_z - log_q_z_x
    return weights

def compute_NLL(weights):
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max()))) + weights.max()) 
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
        
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    assert opt.dataroot == '../../data', 'please go config.py and modify dataroot to "../../data"'
    
    print('Q. IC type?')
    print('(1) png (2) jp2')
    ans = input('-> ')
    if ans == '1':
        opt.ic_type = 'png'
    elif ans == '2':
        opt.ic_type = 'jp2'
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    print('Q. Test Num?')
    test_num = int(input('-> '))

    ######################################################################
    
    #for ood in opt.ood_list:
    for ood in opt.ood_list:

        torch.manual_seed(2021)
        random.seed(2021)
        dataloader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=True,
        )
        
        Complexity = []
        difference = []
        
        for i, x in enumerate(dataloader):
            try:
                x, _ = x
            except:
                pass
            
            with torch.no_grad():
                x = x.to(device)
                _, NLL_loss, _ = model(x, None)
                NLL_loss = NLL_loss.detach().cpu().numpy()[0]
                img = x[0].permute(1, 2, 0)
                img = img.detach().cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                if opt.ic_type == 'jp2':
                    img_encoded = cv2.imencode('.jp2',img)
                elif opt.ic_type == 'png':
                    img_encoded = cv2.imencode('.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                else:
                    assert 0==1
                L = len(img_encoded[1]) * 8
                Complexity.append(L)
                difference.append(NLL_loss - L)
            
                print(f'{ood} GLOW: image {i} IC loss {NLL_loss - L:.2f}')
            
            if i >= test_num - 1:
                break
                
        difference = np.asarray(difference)

        np.save(f'../npy/IC({opt.ic_type})_GLOW/{opt.train_dist}_{ood}.npy', difference)
        print(f'saved {opt.train_dist}_{ood} IC({opt.ic_type})_GLOW npy !')


