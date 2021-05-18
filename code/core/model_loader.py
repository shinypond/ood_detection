import torch
import train_VAE.DCGAN_VAE_pixel as DVAE_pixel
from train_CNN.resnet import ResNet18
from train_CNN.vgg import VGG
import config
import os, sys
from train_GLOW.model import Glow

def load_pretrained_VAE(option='cifar10', ngf=None, nz=None, beta=None, augment='hflip+crop', epoch=200):
    
    """ Load the pre-trained VAE model (for CIFAR10, FMNIST) """
    """ option : 'cifar10' or 'fmnist' is available !! """
    
    if option == 'cifar10':
        opt = config.VAE_cifar10
    elif option == 'fmnist':
        opt = config.VAE_fmnist
    else:
        raise ValueError
        
    if ngf:
        opt.ngf = ngf
    if nz:
        opt.nz = nz
    if beta:
        opt.beta = beta
    
    # 21.05.12 Fixed the final models !
    path_E = f'{opt.modelroot}/VAE_{option}/netE_pixel_ngf_{opt.ngf}_nz_{opt.nz}_beta_{opt.beta:.1f}_augment_{augment}_epoch_{epoch}.pth'
    path_G = f'{opt.modelroot}/VAE_{option}/netG_pixel_ngf_{opt.ngf}_nz_{opt.nz}_beta_{opt.beta:.1f}_augment_{augment}_epoch_{epoch}.pth'
    
    # temp
    #path_E = f'{opt.modelroot}/netE_pixel_cifar10_None_nz200_ngf64_lambda0.pth'
    #path_G = f'{opt.modelroot}/netG_pixel_cifar10_None_nz200_ngf64_lambda0.pth'
    
    
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    netG = DVAE_pixel.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    state_G = torch.load(path_G, map_location=device)
    netG.load_state_dict(state_G)
    netE = DVAE_pixel.Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    state_E = torch.load(path_E, map_location=device)
    netE.load_state_dict(state_E)
    netG.to(device)
    netE.to(device)
    netE = netE.eval()
    netG = netG.eval()
    return netE, netG

def load_pretrained_GLOW(option='cifar10'):
    
    """ Load the pre-trained GlOW model (for CIFAR10, FMNIST???) """
    """ option : 'cifar10' or 'fmnist' is available !! """
    
    if option == 'cifar10':
        opt = config.GLOW_cifar10
        image_shape = (32, 32, 3)
        num_classes = 10
    elif option == 'fmnist':
        opt = config.GLOW_fmnist
        image_shape = (32, 32, 1)
        num_classes = 10
    
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
    )
    
    # trained by BJW (same to "Do generative models know what they don't know")
    model_path = f'{opt.modelroot}/GLOW_{option}/glow_{option}.pt'
    model.load_state_dict(torch.load(model_path)['model']) # there are two keys: 'model', 'optimizer'
    
    # original GLOW model (same to "GLOW" by Kingma)
    # Note : Only available for CIFAR10 (not FMNIST!!)
    #model_path = f'{opt.modelroot}/GLOW_{option}/glow_affine_coupling.pt' 
    #model.load_state_dict(torch.load(model_path))
    
    model.set_actnorm_init()
    model = model.to(device)
    model = model.eval()
    
    return model

def load_pretrained_CNN(option='cifar10', modelname='VGG11'):
    
    """ Load the pre-trained GlOW model (for CIFAR10, FMNIST???) """
    """ option : 'cifar10' or 'fmnist' is available !! """
    """ modelname : 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ResNet18' """
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if option == 'cifar10':
        opt = config.CNN_cifar10
        image_shape = (32, 32, 3)
    elif option == 'fmnist':
        opt = config.CNN_fmnist
        image_shape = (32, 32, 1)
    else:
        raise ValueError
        
    if modelname[:3] == 'VGG':
        model = VGG(modelname, opt.nc).to(device)
    elif modelname == 'ResNet18':
        model = ResNet18(opt.nc).to(device)
    else:
        raise ValueError
    
    # trained by YCY
    #model_path = f'{opt.modelroot}/CNN_{option}/cnn_augment_{augment}_epoch_{epoch}.pth'
    model = torch.nn.DataParallel(model)
    model_path = f'{opt.modelroot}/CNN_{option}/{option}_{modelname}_ckpt.pth'
    model.load_state_dict(torch.load(model_path)['net'])
    model = model.eval()
    
    return model