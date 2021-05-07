
import argparse
import random
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
import os
from custom_loss import VAE_loss, loglikelihood

def KL_div(mu,logvar,reduction = 'avg'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1) 
        return KL


def perturb(x, mu,device):
    b,c,h,w = x.size()
    mask = torch.rand(b,c,h,w)<mu
    mask = mask.float().to(device)
    noise = torch.FloatTensor(x.size()).random_(0, 256).to(device)
    x = x*255
    perturbed_x = ((1-mask)*x + mask*noise)/255.
    return perturbed_x


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help = 'hidden channel size')
    
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=1., help='beta for beta-vae')
    parser.add_argument('--ELBO', type=int, default=1, help='ELBO or Loglikelihood')
    
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    

    opt = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    
    if opt.experiment is None:
        opt.experiment = 'cifar'
        dataset = dset.CIFAR10( root=opt.dataroot, download=True,train = True,
                                transform=transforms.Compose([
                                   transforms.Resize((opt.imageSize)),
                                   transforms.ToTensor(),
                               ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                                shuffle=True, num_workers=int(opt.workers))
        
    if opt.experiment == 'fmnist':
        dataset_fmnist_train = dset.FashionMNIST(root=opt.dataroot, train=True, download=True, transform=transforms.Compose([
                                    transforms.Resize((opt.imageSize)),
                                    transforms.ToTensor(),
                                ]))
        dataloader = torch.utils.data.DataLoader(dataset_fmnist_train, batch_size=opt.batchSize,
                                                shuffle=True, num_workers=int(opt.workers))
        print('dataloader for {} is ready'.format(opt.experiment))
    
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    
    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu)
    netG.apply(weights_init)

    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    netE.apply(weights_init)

    
    netE.to(device)
    netG.to(device)
    
    # setup optimizer
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr)
    netE.train()
    netG.train()
    
    for epoch in range(opt.niter):
        for i, (x, _) in enumerate(dataloader):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            x = x.to(device)
            [z,mu,logvar] = netE(x)
            recon = netG(z)
            
            if opt.ELBO:
                loss = VAE_loss(x, [recon, mu, logvar], beta=opt.beta)
            else:
                loss = loglikelihood(x, z , [recon, mu, logvar])
            loss.backward(retain_graph=True)
            optimizer1.step()
            optimizer2.step()
            
            if not i % 100:
                print('epoch:{} loss : {}'.format(epoch, loss))
    if not os.path.exists('../../saved_models/{}'.format(opt.experiment)):
        os.makedirs('../../saved_models/{}'.format(opt.experiment))
    torch.save(netG.state_dict(), '../../saved_models/{}/netG_ngf{}nz{}beta{}.pth'.format(opt.experiment, ngf, nz, int(10*opt.beta)))
    torch.save(netE.state_dict(), '../../saved_models/{}/netE_ngf{}nz{}beta{}.pth'.format(opt.experiment, ngf, nz, int(10*opt.beta)))