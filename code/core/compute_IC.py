import os, sys, argparse, random, copy, cv2
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

    ######################################################################
    
    for ood in opt.ood_list:

        torch.manual_seed(2021)
        random.seed(2021)
        dataloader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=False,
        )
        
        Complexity = []
        difference = []
        if ood == opt.train_dist:
            start = datetime.now()
        
        for i, x in enumerate(dataloader):
            try:
                x, _ = x
            except:
                pass
            x = x.expand(opt.repeat, -1, -1, -1).contiguous()
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
                NLL_loss = compute_NLL(weights_agg).detach().cpu().numpy()

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
                print(f'{ood} VAE: image {i} IC loss {NLL_loss - L:.2f}')
            
            if i >= test_num - 1:
                break
                
        difference = np.asarray(difference)
        if ood == opt.train_dist:
            end = datetime.now()
            avg_time = (end - start).total_seconds() / test_num

        np.save(f'../npy/ic({opt.ic_type})/{opt.train_dist}_{ood}.npy', difference)
        print(f'saved {opt.train_dist}_{ood} IC(png) npy !')
        
    print(f'average {opt.train_dist} inference time : {avg_time} seconds')
    print(f'average #images processed : {1 / avg_time} images')

    model = 'IC(png)'
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



