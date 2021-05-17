import torch
import DCGAN_VAE_pixel as DVAE
import config

def load_pretrained_VAE(option='cifar10'):
    
    """ Load the pre-trained VAE model (for CIFAR10, FMNIST) """
    """ option : 'cifar10' or 'fmnist' is available !! """
    
    if option == 'cifar10':
        opt = config.VAE_cifar10
    elif option == 'cifar10_150':
        opt = config.VAE_cifar10_150
    elif option == 'cifar10_hflip':
        opt = config.VAE_cifar10_hflip
    else:
        assert option == 'fmnist'
        opt = config.VAE_fmnist
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    netG = DVAE.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    state_G = torch.load(opt.state_G, map_location=device)
    netG.load_state_dict(state_G)
    netE = DVAE.Encoder(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    state_E = torch.load(opt.state_E, map_location=device)
    netE.load_state_dict(state_E)
    netG.to(device)
    netE.to(device)
    netE.eval()
    netG.eval()
    return netE, netG
