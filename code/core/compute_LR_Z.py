import os, sys, argparse, random, copy
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
        log_p_x_z = -torch.sum(cross_entropy.view(b, -1), 1)
        log_p_z = -torch.sum(z**2 / 2 + np.log(2 * np.pi) / 2,1)
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
    
    ##############################################################
    
    for ood in opt.ood_list:
        
        torch.manual_seed(2021)
        random.seed(2021)
            dataloader = TEST_loader(
                train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            )

        NLL_regret = []
        NLL = []
        if ood == opt.train_dist:
            start = datetime.now()
        
        for i, x in enumerate(dataloader):
            try:
                xi, _ = xi
            except:
                pass
            x = xi.expand(opt.repeat,-1,-1,-1).contiguous()
            weights_agg  = []
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
                weights_agg = torch.stack(weights_agg).view(-1) 
                NLL_loss_before = compute_NLL(weights_agg) 
                NLL = np.append(NLL, NLL_loss_before.detach().cpu().numpy())

            xi = xi.to(device)
            b = xi.size(0)
            [z, mu, logvar] = netE(xi)
            mu = nn.Parameter(mu)
            logvar = nn.Parameter(logvar)
            optimizer = optim.Adam([mu, logvar], lr=1e-4, betas=(opt.beta1, 0.999),weight_decay=0.)
            target = Variable(xi.data.view(-1) * 255).long()

            for it in range(opt.num_iter):
                epsilon = torch.randn(z.shape).to(device)
                z = mu + epsilon*((logvar/2).exp())
                recon = netG(z)
                recon = recon.contiguous()
                recon = recon.view(-1,256)
                recl = loss_fn(recon, target)
                recl = torch.sum(recl) / b
                kld = KL_div(mu,logvar)
                loss =  recl + kld.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            weights_agg  = []
            with torch.no_grad():
                xi = xi.expand(opt.repeat,-1,-1,-1).contiguous()
                mu = mu.expand(opt.repeat,-1,-1,-1).contiguous()
                logvar = logvar.expand(opt.repeat,-1,-1,-1).contiguous()
                target = Variable(xi.data.view(-1) * 255).long()
                z = torch.normal(mu, (logvar/2).exp())
                recon = netG(z)
                recon = recon.contiguous()
                mu_ = mu.view(mu.size(0),mu.size(1))
                logvar_ = logvar.view(logvar.size(0), logvar.size(1))
                z_ = z.view(z.size(0),z.size(1))
                weights = store_NLL(xi, recon, mu_, logvar_, z_)
                weights_agg.append(weights)
                weights_agg = torch.stack(weights_agg).view(-1) 
                NLL_loss_after = compute_NLL(weights_agg) 
                regret = NLL_loss_before  - NLL_loss_after
                print(f'{ood} image {i} OPT: {NLL_loss_after.item()} VAE: {NLL_loss_before.item()} diff:{NLL_loss_before.item() - NLL_loss_after.item()}')
                NLL_regret = np.append(NLL_regret, regret.detach().cpu().numpy())
                
            if i >= test_num - 1:
                break

        if ood == opt.train_dist:
            end = datetime.now()
            avg_time = (end - start).total_seconds() / test_num

        np.save(f'../npy/ic({opt.ic_type})/{opt.train_dist}_{ood}.npy', difference)
        print(f'saved {opt.train_dist}_{ood} LR(Z) npy !')
        
    print(f'average {opt.train_dist} inference time : {avg_time} seconds')
    print(f'average #images processed : {1 / avg_time} images')

    model = 'LR(Z)'
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
        print(f'in-dist : {opt.train_dist}, ood : {ood} => AUROC : {auroc}')



                

