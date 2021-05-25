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
import train_VAE.DCGAN_VAE_pixel as DVAE
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
        opt = config.VAE_cifar10
        opt.repeat = 20
        opt.state_E = f'../../saved_models/VAE_cifar10/netE_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_G = f'../../saved_models/VAE_cifar10/netG_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_E_bg = f'../../saved_models/VAE_cifar10/LRatio_BG/netE_bg_ngf_64_nz_200_beta_1.0_augment_hflip_decay_0_epoch_100.pth'
        opt.state_G_bg = f'../../saved_models/VAE_cifar10/LRatio_BG/netG_bg_ngf_64_nz_200_beta_1.0_augment_hflip_decay_0_epoch_100.pth'
    elif ans == '2':
        opt = config.VAE_fmnist
        opt.repeat = 20
        opt.state_E = f'../../saved_models/VAE_fmnist/netE_pixel_ngf_32_nz_100_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_G = f'../../saved_models/VAE_fmnist/netG_pixel_ngf_32_nz_100_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_E_bg = f'../../saved_models/VAE_fmnist/LRatio_BG/netE_bg_ngf_32_nz_100_beta_1.0_augment_hflip_decay_0_epoch_100.pth'
        opt.state_G_bg = f'../../saved_models/VAE_fmnist/LRatio_BG/netG_bg_ngf_32_nz_100_beta_1.0_augment_hflip_decay_0_epoch_100.pth'
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    assert opt.dataroot == '../../data', 'please go config.py and modify dataroot to "../../data"'
    
    print('Q. Test Num?')
    test_num = int(input('-> '))
        
    opt.beta1 = 0.9
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    
    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    state_G = torch.load(opt.state_G, map_location = device)
    netG.load_state_dict(state_G)
    netG_bg = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    state_G_bg = torch.load(opt.state_G_bg, map_location = device)
    netG_bg.load_state_dict(state_G_bg)
    
    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    state_E = torch.load(opt.state_E, map_location = device)
    netE.load_state_dict(state_E)
    netE_bg = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    state_E_bg = torch.load(opt.state_E_bg, map_location = device)
    netE_bg.load_state_dict(state_E_bg)

    netG.to(device)
    netG.eval()
    netG_bg.to(device)
    netG_bg.eval()
    netE.to(device)
    netE.eval()
    netE_bg.to(device)
    netE_bg.eval()
    
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')

    ############################################################################
    
    for ood in opt.ood_list:
        
        torch.manual_seed(2021)
        random.seed(2021)
        dataloader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=False,
        )

        NLL_test_ood = []
        NLL_test_ood_bg = []
        if ood == opt.train_dist:
            start = datetime.now()
        
        for i, x in enumerate(dataloader):
            try:
                x, _ = x
            except:
                pass
            x = x.expand(opt.repeat, -1, -1, -1).contiguous()
            weights_agg  = []
            weights_agg_bg = []
            with torch.no_grad():
                x = x.to(device)
                b = x.size(0)

                [z, mu, logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0), mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0), z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z)
                weights_agg.append(weights)

                [z_bg, mu_bg, logvar_bg] = netE_bg(x)
                recon_bg = netG_bg(z_bg)
                mu_bg = mu_bg.view(mu_bg.size(0), mu_bg.size(1))
                logvar_bg = logvar_bg.view(logvar_bg.size(0), logvar_bg.size(1))
                z_bg = z_bg.view(z_bg.size(0), z_bg.size(1))
                weights_bg = store_NLL(x, recon_bg, mu_bg, logvar_bg, z_bg)
                weights_agg_bg.append(weights_bg)

                weights_agg = torch.stack(weights_agg).view(-1) 
                weights_agg_bg = torch.stack(weights_agg_bg).view(-1)

                NLL_loss = compute_NLL(weights_agg) 
                NLL_loss_bg =  compute_NLL(weights_agg_bg) 

                NLL_test_ood.append(NLL_loss.detach().cpu().numpy())
                NLL_test_ood_bg.append(NLL_loss_bg.detach().cpu().numpy())
                diff = -NLL_loss.item() + NLL_loss_bg.item()
                print(f'{ood} VAE: image {i} NLL {NLL_loss.item()}, NLL BG {NLL_loss_bg.item()}, diff: {diff}')
                
            if i >= test_num - 1:
                break

        NLL_test_ood = np.asarray(NLL_test_ood)
        NLL_test_ood_bg = np.asarray(NLL_test_ood_bg)
        metric_ood = -NLL_test_ood + NLL_test_ood_bg
        
        if ood == opt.train_dist:
            end = datetime.now()
            avg_time = (end - start).total_seconds() / test_num
        
        np.save(f'../npy/LRatio/{opt.train_dist}_{ood}.npy', metric_ood)
        print(f'SAVED {opt.train_dist}_{ood} LRatio npy !')
        
    print(f'Average {opt.train_dist} Inference Time : {avg_time} seconds')
    print(f'Average #Images Processed : {1 / avg_time} Images')

    model = 'LRatio'
    print(f'{model} / {opt.train_dist}')
    indist = np.load(f'../npy/{model}/{opt.train_dist}_{opt.train_dist}.npy')
    label1 = np.ones(len(indist))
    
    for ood in opt.ood_list:
        ood_ = np.load(f'../npy/{model}/{opt.train_dist}_{ood}.npy')
        combined = np.concatenate((indist, ood_))
        label2 = np.ones(len(ood_))
        label = np.concatenate((label1, label2))
        fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
        auroc = metrics.auc(fpr, tpr)
        print(f'In-dist : {opt.train_dist}, OOD : {ood} => AUROC : {1 - auroc}')



        