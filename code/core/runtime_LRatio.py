import os, sys, argparse, random, copy
import numpy as np
from datetime import datetime
from datetime import timedelta

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../data', help='path to dataset')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    
    parser.add_argument('--state_E', default='../saved_models/VAE_cifar10/netE_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G', default='../saved_models/VAE_cifar10/netG_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth', help='path to encoder checkpoint')

    parser.add_argument('--state_E_bg', default='../saved_models/VAE_cifar10/LRatio/netE_bg_ngf_64_nz_200_beta_1.0_augment_hflip_decay_0_epoch_100.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G_bg', default='../saved_models/VAE_cifar10/LRatio/netG_bg_ngf_64_nz_200_beta_1.0_augment_hflip_decay_0_epoch_100.pth', help='path to encoder checkpoint')

    opt = parser.parse_args()
    
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
        assert 0==1, 'Not Yet Prepared the pre-trained VAE model (Background)'
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

    """
    ########################################################################
    id_dataloader = TEST_loader(
        train_dist=opt.train_dist,
        target_dist=opt.train_dist,
        shuffle=True,
        is_glow=False,
    )
    NLL_test_indist = []
    NLL_test_indist_bg = []
    start = datetime.now()

    for i, (x, _) in enumerate(id_dataloader):
        x = x.expand(opt.repeat,-1,-1,-1).contiguous()
        weights_agg  = []
        weights_agg_bg = []
        with torch.no_grad():
            
            x = x.to(device)
            b = x.size(0)

            [z,mu,logvar] = netE(x)
            recon = netG(z)
            mu = mu.view(mu.size(0),mu.size(1))
            logvar = logvar.view(logvar.size(0), logvar.size(1))
            z = z.view(z.size(0),z.size(1))
            weights = store_NLL(x, recon, mu, logvar, z)
            weights_agg.append(weights)

            [z_bg,mu_bg,logvar_bg] = netE_bg(x)
            recon_bg = netG_bg(z_bg)
            mu_bg = mu_bg.view(mu_bg.size(0),mu_bg.size(1))
            logvar_bg = logvar_bg.view(logvar_bg.size(0), logvar_bg.size(1))
            z_bg = z_bg.view(z_bg.size(0),z_bg.size(1))
            weights_bg = store_NLL(x, recon_bg, mu_bg, logvar_bg, z_bg)

            weights_agg_bg.append(weights_bg)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            weights_agg_bg = torch.stack(weights_agg_bg).view(-1)
           
            NLL_loss = compute_NLL(weights_agg) 
            NLL_loss_bg =  compute_NLL(weights_agg_bg) 
            NLL_test_indist.append(NLL_loss.detach().cpu().numpy())
            NLL_test_indist_bg.append(NLL_loss_bg.detach().cpu().numpy())
            diff = -NLL_loss.item() + NLL_loss_bg.item()
            print('Indist VAE: image {} NLL {}, NLL BG {}, diff {}'.format(i, NLL_loss.item(),NLL_loss_bg.item(), diff))
            
        if i >= test_num:
            break
    
    NLL_test_indist = np.asarray(NLL_test_indist)
    NLL_test_indist_bg = np.asarray(NLL_test_indist_bg)
    metric_indist = -NLL_test_indist + NLL_test_indist_bg
    end = datetime.now()
    acc_time = end - start
    acc_time = acc_time.total_seconds()
    print(f'Average Inference Time {acc_time / test_num}')
    print(f'Processed Images per Second {test_num / acc_time:.2f}')
    
    np.save(f'../npy/LRatio/{opt.train_dist}_{opt.train_dist}.npy', metric_indist)
    print('SAVED In-dist LRatio npy !')
    
    ###################################################################################
    """
    dataloaders = []
    for ood in opt.ood_list:
        if ood != opt.train_dist:
            loader = TEST_loader(
                train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            )
            dataloaders.append(loader)
    
    NLL_test_ood = []
    NLL_test_ood_bg = []
    loader_idx_list = np.random.randint(0, len(dataloaders), size=test_num)
    i = 0

    while i < test_num:
        x = next(iter(dataloaders[loader_idx_list[i]]))
        ood = opt.ood_list[loader_idx_list[i]+1]
        print(f'Selected OOD : {ood}')
        try:
            x, _ = x
        except:
            pass
        x = x.expand(opt.repeat,-1,-1,-1).contiguous()
        weights_agg  = []
        weights_agg_bg = []
        with torch.no_grad():
            x = x.to(device)
            b = x.size(0)

            [z,mu,logvar] = netE(x)
            recon = netG(z)
            mu = mu.view(mu.size(0),mu.size(1))
            logvar = logvar.view(logvar.size(0), logvar.size(1))
            z = z.view(z.size(0),z.size(1))
            weights = store_NLL(x, recon, mu, logvar, z)
            weights_agg.append(weights)

            [z_bg,mu_bg,logvar_bg] = netE_bg(x)
            recon_bg = netG_bg(z_bg)
            mu_bg = mu_bg.view(mu_bg.size(0),mu_bg.size(1))
            logvar_bg = logvar_bg.view(logvar_bg.size(0), logvar_bg.size(1))
            z_bg = z_bg.view(z_bg.size(0),z_bg.size(1))
            weights_bg = store_NLL(x, recon_bg, mu_bg, logvar_bg, z_bg)
            weights_agg_bg.append(weights_bg)
            
            weights_agg = torch.stack(weights_agg).view(-1) 
            weights_agg_bg = torch.stack(weights_agg_bg).view(-1)
           
            NLL_loss = compute_NLL(weights_agg) 
            NLL_loss_bg =  compute_NLL(weights_agg_bg) 
            
            NLL_test_ood.append(NLL_loss.detach().cpu().numpy())
            NLL_test_ood_bg.append(NLL_loss_bg.detach().cpu().numpy())
            diff = -NLL_loss.item() + NLL_loss_bg.item()
            print('MNIST VAE: image {} NLL {}, NLL BG {}, diff: {}'.format(i, NLL_loss.item(),NLL_loss_bg.item(), diff))
        i += 1
        
    NLL_test_ood = np.asarray(NLL_test_ood)
    NLL_test_ood_bg = np.asarray(NLL_test_ood_bg)
    metric_ood = -NLL_test_ood + NLL_test_ood_bg
    np.save(f'../npy/LRatio/{opt.train_dist}_random_ood.npy', metric_ood)
    print('SAVED OOD LRatio npy !')