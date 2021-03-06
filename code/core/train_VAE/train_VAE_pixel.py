import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
import os, sys
#from pathlib import Path
#path = Path(os.getcwd()).parent.parent / 'core'
#sys.path.append(str(path))
import DCGAN_VAE_pixel as DVAE

def plot_loss(dataset_name, history, epoch):
    
    x = range(len(history))
    y = np.array(history)
    fig = plt.figure(figsize=(16, 9))
    plt.plot(x, y, label='total loss')
    plt.grid()
    plt.legend()
    fig.savefig(f'./temp/{dataset_name}_loss_graph_epoch_{epoch}.png')
    

def KL_div(mu,logvar,reduction = 'avg'):
    mu = mu.view(mu.size(0),mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1) 
        return KL

def perturb(x, mu, device):
    b, c, h, w = x.size()
    mask = torch.rand(b, c, h, w) < mu
    mask = mask.float().to(device)
    noise = torch.FloatTensor(x.size()).random_(0, 256).to(device)
    x = x * 255
    perturbed_x = ((1 - mask) * x + mask * noise)/255.
    return perturbed_x

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../../../data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help = 'hidden channel sieze')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=1., help='beta for beta-vae')
    
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    #parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--perturbed', action=None, help='Whether to train on perturbed data, used for comparing with likelihood ratio by Ren et al.')
    parser.add_argument('--ratio', type=float, default=0.2, help='ratio for perturbation of data, see Ren et al.')
    
    opt = parser.parse_args()
    opt.dataroot = '../../../data'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    
    
    
#############################################################
#############################################################
#############################################################

    # Ask train-dataset (1) CIFAR-10 (2) FMNIST 
    
    traindist = input('Q. Which dataset do you want to train the model for??\n(1) CIFAR-10 (2) FMNIST (Press 1 or 2)\n')
    augment = input('Q. Apply data augmentation?\n(1) None (2) hflip (3) hflip + Crop (Press 1 or 2 or 3)\n')
    
    if augment == '1':
        opt.augment = 'None'
        transform = transforms.Compose([
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ])
    elif augment == '2':
        opt.augment = 'hflip'
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ])
    elif augment == '3':
        opt.augment = 'hflip+crop'
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ])
        # if you want to add other augmentations, then append them at this point!
    else:
        raise ValueError('Oops! Please insert 1 or 2 or 3. Bye~')
    
    if traindist == '1':
        opt.nc = 3
        opt.niter = 200
        opt.ngf = 64
        opt.train_dist = 'cifar10'
        experiment = '../../../saved_models/VAE_cifar10'
        dataset = dset.CIFAR10(
            root=opt.dataroot,
            download=False,
            train=True,
            transform=transform,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
        )
    elif traindist == '2':
        opt.nc = 1
        opt.niter = 100
        opt.ngf = 32
        opt.train_dist = 'fmnist'
        experiment = '../../../saved_models/VAE_fmnist'
        dataset = dset.FashionMNIST(
            root=opt.dataroot,
            download=False,
            train=True,
            transform=transform,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers)
        )
    else:
        raise ValueError('Oops! Please insert 1 or 2. Bye~')
        
    name = experiment.split('/')[-1] # This is 'VAE_cifar10' or 'VAE_fmnist'
    print(f'Dataloader for {name} is ready !')
    print(f'Please see the path "{experiment}" for the saved model !')
    
#############################################################
#############################################################
#############################################################
    
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    print(f'Channel {nc}, ngf {ngf}, nz {nz}, augment {opt.augment}')
    beta = opt.beta
    
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
    
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay=0)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=0)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.5)

    netE.train()
    netG.train()

    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    rec_l = []
    kl = []
    history = []
    start = datetime.today()
    for epoch in range(opt.niter):
        mean_loss = 0.0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            if opt.perturbed:
                x = perturb(x, opt.ratio, device)

            b = x.size(0)
            target = Variable(x.data.view(-1) * 255).long()
            [z, mu, logvar] = netE(x)
            recon = netG(z)
            
            recon = recon.contiguous()
            recon = recon.view(-1, 256)
            recl = loss_fn(recon, target)
            recl = torch.sum(recl) / b
            kld = KL_div(mu, logvar)
            
            loss = recl + opt.beta * kld.mean()
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss = loss
            loss.backward(retain_graph=True)
            

            optimizer1.step()
            optimizer2.step()
            rec_l.append(recl.detach().item())
            kl.append(kld.mean().detach().item())
            mean_loss = (mean_loss * i + loss.detach().item()) / (i + 1)
        
            if not i % 100:
                print(f'epoch:{epoch} recon:{np.mean(rec_l):.6f} kl:{np.mean(kl):.6f}')
        
        history.append(mean_loss)
        scheduler1.step()
        scheduler2.step()
        now = datetime.today()
        print(f'\nNOW : {now:%Y-%m-%d %H:%M:%S}, Elapsed Time : {now - start}\n')
        if epoch % 50 == 49:
            torch.save(netE.state_dict(), experiment + f'/netE_pixel_ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_augment_{opt.augment}_epoch_{epoch+1}.pth')
            torch.save(netG.state_dict(), experiment + f'/netG_pixel_ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_augment_{opt.augment}_epoch_{epoch+1}.pth')
            
    torch.save(netE.state_dict(), experiment + f'/netE_pixel_ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_augment_{opt.augment}_epoch_{opt.niter}.pth')
    torch.save(netG.state_dict(), experiment + f'/netG_pixel_ngf_{ngf}_nz_{nz}_beta_{beta:.1f}_augment_{opt.augment}_epoch_{opt.niter}.pth')
    

if __name__=="__main__":
    main()
