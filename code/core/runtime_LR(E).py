import os, sys, argparse, random, copy
import numpy as np
from datetime import datetime
from datetime import timedelta

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
import train_VAE.DCGAN_VAE_pixel as DVAE
import config

# fix a random seed
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

def KL_div(mu, logvar, reduction = 'none'):
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) 
        return KL

def store_NLL(x, recon, mu, logvar, z):
    with torch.no_grad():
        sigma = torch.exp(0.5 * logvar)
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        recon = recon.contiguous()
        recon = recon.view(-1, 256)
        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b , -1), 1)
        log_p_z = -torch.sum(z**2 / 2 + np.log(2 * np.pi) / 2, 1)
        z_eps = (z - mu) / sigma
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
        opt = config.VAE_cifar10
        opt.repeat = 20
        opt.state_E = f'../../saved_models/VAE_cifar10/netE_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_G = f'../../saved_models/VAE_cifar10/netG_pixel_ngf_64_nz_200_beta_1.0_augment_hflip_epoch_100.pth'
    elif ans == '2':
        opt = config.VAE_fmnist
        opt.repeat = 20
        opt.state_E = f'../../saved_models/VAE_fmnist/netE_pixel_ngf_32_nz_100_beta_1.0_augment_hflip_epoch_100.pth'
        opt.state_G = f'../../saved_models/VAE_fmnist/netG_pixel_ngf_32_nz_100_beta_1.0_augment_hflip_epoch_100.pth'
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    assert opt.dataroot == '../../data', 'please go config.py and modify dataroot to "../../data"'
    
    print('Q. Test Num?')
    test_num = int(input('-> '))
    
    opt.num_iter = 100
    opt.beta1 = 0.9
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

###############################################################
    id_dataloader = TEST_loader(
        train_dist=opt.train_dist,
        target_dist=opt.train_dist,
        shuffle=True,
        is_glow=False,
    )
        
    NLL_regret = []
    NLL = []
    i = 0
    while i < test_num:
        xi = next(iter(id_dataloader))
        try:
            xi, _ = xi
        except:
            pass
        x = xi.expand(opt.repeat, -1, -1, -1).contiguous()
        weights_agg  = []
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
            weights_agg = torch.stack(weights_agg).view(-1) 
            NLL_loss_before = compute_NLL(weights_agg) 
            NLL = np.append(NLL, NLL_loss_before.detach().cpu().numpy())

        xi = xi.to(device)
        b = xi.size(0)
        netE_copy = copy.deepcopy(netE)
        netE_copy.eval()
        optimizer = optim.Adam(netE_copy.parameters(), lr=1e-4, betas=(opt.beta1, 0.999),weight_decay=5e-5) ## Double Check needed
        target = Variable(xi.data.view(-1) * 255).long()

        for it in range(opt.num_iter):    
            [z,mu,logvar] = netE_copy(xi)
            recon = netG(z)
            recon = recon.contiguous()
            recon = recon.view(-1, 256)
            recl = loss_fn(recon, target)
            recl = torch.sum(recl) / b
            kld = KL_div(mu,logvar)
            loss =  recl + kld.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        weights_agg  = []
        with torch.no_grad():
            xi = xi.expand(opt.repeat, -1, -1, -1).contiguous()
            target = Variable(xi.data.view(-1) * 255).long()
            [z, mu, logvar] = netE_copy(xi)
            recon = netG(z)
            recon = recon.contiguous()
            mu = mu.view(mu.size(0), mu.size(1))
            logvar = logvar.view(logvar.size(0), logvar.size(1))
            z = z.view(z.size(0), z.size(1))
            weights = store_NLL(xi, recon, mu, logvar, z)
            weights_agg.append(weights)
            weights_agg = torch.stack(weights_agg).view(-1) 
            NLL_loss_after = compute_NLL(weights_agg) 
            regret = NLL_loss_before - NLL_loss_after
            print(f'{opt.train_dist} image {i} OPT: {NLL_loss_after.item()} VAE: {NLL_loss_before.item()} diff: {NLL_loss_before.item() - NLL_loss_after.item()}')
            NLL_regret = np.append(NLL_regret, regret.detach().cpu().numpy())
        i += 1

    np.save(f'../npy/LR(E)/{opt.train_dist}_{opt.train_dist}.npy', NLL_regret)
    print(f'SAVED In-dist REGRET npy !')
###############################################################
    dataloaders = []
    for ood in opt.ood_list:
        loader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=False,
        )
        dataloaders.append(loader)
        
    NLL_regret = []
    NLL = []
    loader_idx_list = np.random.randint(0, len(dataloaders), size=test_num)
    acc_time = timedelta(0)
    i = 0
    while i < test_num:
        xi = next(iter(dataloaders[loader_idx_list[i]]))
        ood = opt.ood_list[loader_idx_list[i]]
        print(f'Selected OOD : {ood}')
        try:
            xi, _ = xi
        except:
            pass
        start = datetime.now()
        x = xi.expand(opt.repeat, -1, -1, -1).contiguous()
        weights_agg  = []
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
            weights_agg = torch.stack(weights_agg).view(-1) 
            NLL_loss_before = compute_NLL(weights_agg) 
            NLL = np.append(NLL, NLL_loss_before.detach().cpu().numpy())

        xi = xi.to(device)
        b = xi.size(0)
        netE_copy = copy.deepcopy(netE)
        netE_copy.eval()
        optimizer = optim.Adam(netE_copy.parameters(), lr=1e-4, betas=(opt.beta1, 0.999),weight_decay=5e-5) ## Double Check needed
        target = Variable(xi.data.view(-1) * 255).long()

        for it in range(opt.num_iter):    
            [z,mu,logvar] = netE_copy(xi)
            recon = netG(z)
            recon = recon.contiguous()
            recon = recon.view(-1, 256)
            recl = loss_fn(recon, target)
            recl = torch.sum(recl) / b
            kld = KL_div(mu,logvar)
            loss =  recl + kld.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        weights_agg  = []
        with torch.no_grad():
            xi = xi.expand(opt.repeat, -1, -1, -1).contiguous()
            target = Variable(xi.data.view(-1) * 255).long()
            [z, mu, logvar] = netE_copy(xi)
            recon = netG(z)
            recon = recon.contiguous()
            mu = mu.view(mu.size(0), mu.size(1))
            logvar = logvar.view(logvar.size(0), logvar.size(1))
            z = z.view(z.size(0), z.size(1))
            weights = store_NLL(xi, recon, mu, logvar, z)
            weights_agg.append(weights)
            weights_agg = torch.stack(weights_agg).view(-1) 
            NLL_loss_after = compute_NLL(weights_agg) 
            regret = NLL_loss_before - NLL_loss_after
            end = datetime.now()
            acc_time += (end - start)
            print(f'{ood} image {i} OPT: {NLL_loss_after.item()} VAE: {NLL_loss_before.item()} diff: {NLL_loss_before.item() - NLL_loss_after.item()} Time {end - start}')
            NLL_regret = np.append(NLL_regret, regret.detach().cpu().numpy())
        i += 1

    acc_time = acc_time.total_seconds()
    print(f'Average Inference Time {acc_time / test_num}')
    print(f'Processed Images per Second {test_num / acc_time:.2f}')
    
    np.save(f'../npy/LR(E)/{opt.train_dist}_random_ood.npy', NLL_regret)
    print(f'SAVED OOD REGRET npy !')

