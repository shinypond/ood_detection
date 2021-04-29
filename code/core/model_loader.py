import torch
import train_VAE.DCGAN_VAE_pixel as DVAE
import config
import os, sys
from train_GLOW.model import Glow

def load_pretrained_VAE(option='cifar10', ngf=None, nz=None, beta=None, augment=None):
    
    """ Load the pre-trained VAE model (for CIFAR10, FMNIST) """
    """ option : 'cifar10' or 'fmnist' is available !! """
    """ ngf, nz, beta : You can choose them !! There are various pre-trained VAE models ! """
    
    if option == 'cifar10':
        opt = config.VAE_cifar10
    else:
        assert option == 'fmnist'
        opt = config.VAE_fmnist
        
    if ngf:
        opt.ngf = ngf
    if nz:
        opt.nz = nz
    if beta:
        opt.beta = beta
    
    if augment == None:
        augment = 'None'
        
    path_E = f'{opt.modelroot}/VAE_{option}/netE_ngf_{opt.ngf}_nz_{opt.nz}_beta_{opt.beta:.1f}_augment_{augment}.pth'
    path_G = f'{opt.modelroot}/VAE_{option}/netG_ngf_{opt.ngf}_nz_{opt.nz}_beta_{opt.beta:.1f}_augment_{augment}.pth'
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    netG = DVAE.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    state_G = torch.load(path_G, map_location=device)
    netG.load_state_dict(state_G)
    netE = DVAE.Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    state_E = torch.load(path_E, map_location=device)
    netE.load_state_dict(state_E)
    netG.to(device)
    netE.to(device)
    netE.eval()
    netG.eval()
    return netE, netG

def load_pretrained_GLOW(option='cifar10'):
    
    """ Load the pre-trained GlOW model (for CIFAR10, FMNIST???) """
    """ option : 'cifar10' or 'fmnist' (???) is available !! """
    
    if option == 'cifar10':
        opt = config.GLOW_cifar10
    else:
        assert option == 'fmnist'
        opt = config.GLOW_fmnist # 에러날 거
    
    image_shape = (32, 32, 3)
    num_classes = 10
    device = torch.device('cuda')

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
    )

    model_path = f'{opt.modelroot}/GLOW_{option}/glow_{option}.pt'
    model.load_state_dict(torch.load(model_path))
    model.set_actnorm_init()
    model = model.to(device)
    model = model.eval()
    
    return model
