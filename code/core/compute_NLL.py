import argparse
import numpy as np
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
import copy
from torchvision.datasets import ImageFolder
import os
import sys
import train_VAE.DCGAN_VAE_pixel as DVAE
from data_loader import TRAIN_loader, TEST_loader
import config
from datetime import datetime


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (img,index)


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
        sigma = torch.exp(0.5*logvar)
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1,256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)
      
        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = (z - mu)/sigma
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
        opt = config.VAE_cifar10
        opt.repeat = 200
        opt.state_E = f'../../saved_models/VAE_cifar10/netE_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_G = f'../../saved_models/VAE_cifar10/netG_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth'
    elif ans == '2':
        opt = config.VAE_fmnist
        opt.repeat = 200
        opt.state_E = f'../../saved_models/VAE_fmnist/netE_pixel_ngf_32_nz_100_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_G = f'../../saved_models/VAE_fmnist/netG_pixel_ngf_32_nz_100_beta_1.0_augment_hflip_epoch_100.pth'
    else:
        raise ValueError('Insert 1 or 2. Bye.')
    
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    state_G = torch.load(opt.state_G, map_location = device)
    netG.load_state_dict(state_G)
    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    state_E = torch.load(opt.state_E, map_location = device)
    netE.load_state_dict(state_E)
    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    start = datetime.now()
    
    for ood in opt.ood_list:
        if ood in ['trafficsign', 'noise', 'constant']:
            continue
        test_loader = TEST_loader(opt.train_dist, ood, shuffle=True, is_glow=False, normalize=False)
        NLL = []
        for i, xi in enumerate(test_loader):
            try: 
                xi, _ = xi
            except:
                pass
            x = xi.expand(opt.repeat,-1,-1,-1).contiguous()
            weights_agg = []
            with torch.no_grad():
                for batch_number in range(5):
                    x = x.to(device)
                    b = x.size(0)
                    [z,mu,logvar] = netE(x)
                    recon = netG(z)
                    mu = mu.view(mu.size(0),mu.size(1))
                    logvar = logvar.view(logvar.size(0), logvar.size(1))
                    z = z.view(z.size(0),z.size(1))
                    weights = store_NLL(x, recon, mu, logvar, z)
                    weights_agg.append(weights)
                weights_agg = torch.stack(weights_agg).view(-1) 
                NLL_loss_before = compute_NLL(weights_agg) 
                NLL = np.append(NLL, NLL_loss_before.detach().cpu().numpy())
            print(f'[ {i+1:04d} / 5000 ] {opt.train_dist}/{ood} NLL running_mean {NLL.mean():.2f} Elapsed time {datetime.now() - start}')
            if i >= 4999:
                break

        np.save(f'./array/{opt.train_dist}_{ood}_nll.npy', NLL)

    