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
import DCGAN_VAE_pixel as DVAE
import torch.nn.functional as F
import copy
from torchvision.datasets import ImageFolder
import os
import sys
sys.path.append(os.getcwd() + '/core')
from core.data_loader import TRAIN_loader, TEST_loader
import core.config as config


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--modelroot', default='../saved_models', help='path to model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--experiment', default='fmnist', help='fmnist or cifar10')
    parser.add_argument('--model', default='VAE', help='VAE or GLOW')
    parser.add_argument('--num_iter', default=100, help='number of iteration')
    parser.add_argument('--repeat', default=20, help='number of MC sampling')
    
    opt = parser.parse_args()
    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    opt.batch_size = 1
    opt.train_batchsize = 1
    opt.beta = 1

    if opt.experiment == 'fmnist':
        opt.nc = 1
        opt.nz = 100
        opt.ngf = 32
        opt.ood_list = ['cifar10', 'svhn', 'celeba', 'lsun', 'mnist', 'fmnist', 'kmnist', 'omniglot', 'notmnist', 'noise', 'constant']

    elif opt.experiment == 'cifar10':
        opt.nc = 3
        opt.nz = 200
        opt.ngf = 64
        opt.ood_list = ['cifar10', 'svhn']#'celeba', 'lsun', 'cifar100', 'mnist', 'fmnist', 'kmnist', 'omniglot', 'notmnist', 'trafficsign', 'noise', 'constant']

    opt.state_E = f'{opt.modelroot}/VAE_{opt.experiment}/netE_pixel_ngf_{opt.ngf}_nz_{opt.nz}_beta_1.0_augment_hflip_epoch_100.pth'
    opt.state_G = f'{opt.modelroot}/VAE_{opt.experiment}/netG_pixel_ngf_{opt.ngf}_nz_{opt.nz}_beta_1.0_augment_hflip_epoch_100.pth'

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)

    print('Building models...')
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

    print('Building complete...')

    for ood in opt.ood_list:
        dataloader = TEST_loader(
                                train_dist=opt.experiment,
                                target_dist=ood,
                                shuffle=True,
                                is_glow=False,
                                )

        NLL_regret = []
        NLL = []

        for i, xi in enumerate(dataloader):
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
            netE_copy = copy.deepcopy(netE)
            netE_copy.eval()
            optimizer = optim.Adam(netE_copy.parameters(), lr=1e-4, betas=(opt.beta1, 0.999),weight_decay=5e-5) ## Double Check needed
            target = Variable(xi.data.view(-1) * 255).long()

            for it in range(opt.num_iter):    
                [z,mu,logvar] = netE_copy(xi)
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
                target = Variable(xi.data.view(-1) * 255).long()
                [z,mu,logvar] = netE_copy(xi)
                recon = netG(z)
                recon = recon.contiguous()
                mu = mu.view(mu.size(0),mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0),z.size(1))
                weights = store_NLL(xi, recon, mu, logvar, z)

                weights_agg.append(weights)
                
                weights_agg = torch.stack(weights_agg).view(-1) 
                NLL_loss_after = compute_NLL(weights_agg) 
                print(f'{ood} image {i} OPT: {NLL_loss_after.item()} VAE: {NLL_loss_before.item()} diff:{NLL_loss_before.item()  - NLL_loss_after.item()}')
                regret = NLL_loss_before  - NLL_loss_after
                NLL_regret = np.append(NLL_regret, regret.detach().cpu().numpy())
            if i >= 500: #test for 500 samples
                break


        np.save(f'./array/nll/{opt.experiment}_{ood}_nll.npy', NLL)
        np.save(f'./array/LR(E)/{opt.experiment}_{ood}_regret.npy', NLL_regret)