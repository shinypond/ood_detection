import numpy as np
from tqdm import tqdm
# If you use jupyter lab, then install @jupyter-widgets/jupyterlab-manager and restart server

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from custom_loss import VAE_loss, loglikelihood
import config
from data_loader import TRAIN_loader, TEST_loader


def _Noise(x, sigma):
    noise = sigma * torch.randn(x.shape)
    if x.is_cuda:
        noise = noise.cuda()
    return x + noise


def Calculate_fisher_VAE(
    netE,
    netG,
    dataloader,
    dicts,
    opt,
    max_iter,
    loss_type='ELBO',
    noise=0.02):
    
    """ netE, netG : Encoder, Decoder of trained VAE """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ dicts : Which LAYERs do you want to see for calculating Fisher ? """
    """ opt : Refer to config.py """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ loss_type : There are two options """
    """     (1) 'ELBO' (traditional VAE loss) """
    """     (2) 'exact' (exact LogLikelihood for VAE ; Jae-moo Choi's NEW(?) viewpoint)"""
    """ noise : perturb input tensor with a small noise ! """
    
    if opt.ngpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    Fisher_inv = {}
    count = 0
    
    for i, (x, _) in enumerate(tqdm(dataloader, desc='Calculate Fisher VAE', unit='step')):
        
        #########################
        ##### Get Gradients #####
        #########################
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
        x_perturb = _Noise(x, noise)
        [z, mu, logvar] = netE(x_perturb)
        
        if opt.num_samples == 1:
            recon = netG(mu)
        elif opt.num_samples > 1:
            recon = netG(z)
        else:
            raise ValueError
        
        recon = recon.contiguous()
        
        if loss_type == 'ELBO':
            loss = VAE_loss(x, [recon, mu, logvar])
        elif loss_type == 'exact':
            loss = loglikelihood(x, z, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
        
        ####################################
        ##### Calculate Fisher Inverse #####
        ####################################
        
        grads = {}
        count += 1
        for param in dicts:
            grads[param] = []
            for j in range(param.grad.shape[0]):
                grads[param].append(param.grad[j, :, :, :].view(-1, 1))
            grads[param] = torch.cat(grads[param], dim=1).T.to(device) # 200 x 4096
            # try
            grads[param] = grads[param].reshape(grads[param].shape[0] * 16, -1) # 3200 x 256
            
            if i == 0:
                Fisher_inv[param] = 1000 * torch.diag(torch.ones(grads[param].shape[1])).unsqueeze(0).to(device) # 1 x 256 x 256
                Fisher_inv[param] = Fisher_inv[param].repeat(grads[param].shape[0], 1, 1) # 3200 x 256 x 256
                
            u1 = grads[param].unsqueeze(1) # 3200 x 1 x 256
            u2 = grads[param].unsqueeze(2) # 3200 x 256 x 1
            b = torch.bmm(Fisher_inv[param], u2)
            denom = torch.ones(grads[param].shape[0], 1).to(device) + torch.bmm(u1, b).squeeze(2) # 3200 x 1
            denom = denom.unsqueeze(2) # 3200 x 1 x 1
            numer = torch.bmm(b, b.permute(0, 2, 1)) # 3200 x 256 x 256
            Fisher_inv[param] -= numer / denom # 3200 x 256 x 256
        
        if i >= max_iter - 1:
            break
        
    ################################
    ##### Normalize Fisher_inv #####
    ################################
        
    normalize_factor = {}
    for param in dicts:
        Fisher_inv[param] *= count
        
        # just try.... (hmm... How about seeing some small part of the Fisher_inv?)
        entry_want2see = (torch.sum((torch.abs(Fisher_inv[param]) >= 0), dim=[1, 2]) != 0) # 3200
        Fisher_inv[param] = Fisher_inv[param][entry_want2see, :, :]
        
        normalize_factor[param] = 2 * np.sqrt(np.array(Fisher_inv[param].shape).prod())
                
    return Fisher_inv, normalize_factor, entry_want2see


def Calculate_score_VAE(
    netE,
    netG,
    dataloader,
    dicts,
    opt,
    Fisher_inv,
    normalize_factor,
    entry_want2see,
    max_iter,
    loss_type='ELBO',
    noise=0.02):
    
    """ netE, netG : Encoder, Decoder of trained VAE """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ dicts : Which LAYERs do you want to see for calculating Fisher ? """
    """ opt : Refer to config.py """
    """ Fisher_inv, normalize_factor : Outputs from the function 'Calculate_fisher_VAE' """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ loss_type : There are two options """
    """     (1) 'ELBO' (traditional VAE loss) """
    """     (2) 'exact' (exact LogLikelihood for VAE ; Jae-moo Choi's NEW(?) viewpoint)"""
    """ noise : perturb input tensor with a small noise ! """
    
    if opt.ngpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
        
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    score = {}
        
    for i, x in enumerate(tqdm(dataloader, desc='Calculate Score VAE', unit='step')):
        
        try: # with label (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # without label (ex : celeba)
            pass
        
        #########################
        ##### Get Gradients #####
        #########################
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
        x_perturb = _Noise(x, noise)
        gradient_val = 0
        [z, mu, logvar] = netE(x)
        
        if opt.num_samples == 1:
            recon = netG(mu)
        elif opt.num_samples > 1:
            recon = netG(z)
        else:
            raise ValueError
            
        recon = recon.contiguous()
        
        if loss_type == 'ELBO':
            loss = VAE_loss(x, [recon, mu, logvar])
        elif loss_type == 'exact':
            loss = loglikelihood(x, z, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)

        #######################
        ##### Get Score ! #####
        #######################
        
        grads = {}
        for param in dicts:
            grads[param] = []
            for j in range(param.grad.shape[0]):
                grads[param].append(param.grad[j, :, :, :].view(-1, 1)) # 1 x 4096
            grads[param] = torch.cat(grads[param], dim=1).T.to(device) # 200 x 4096
            # This is to prevent CUDA Memory Explosion!
            grads[param] = grads[param].reshape(grads[param].shape[0] * 16, -1) # 3200 x 256
            
            # try
            grads[param] = grads[param][entry_want2see, :] # ???? x 256
            
            u1 = grads[param].unsqueeze(1) # 3200 x 1 x 256
            u2 = grads[param].unsqueeze(2) # 3200 x 256 x 1
            
            s = torch.bmm(torch.bmm(u1, Fisher_inv[param]), u2) # 3200 x 1 x 1
            s = s.squeeze(1).squeeze(1) # 3200
            s = torch.sum(s).detach().cpu().numpy()
            if i == 0:
                score[param] = []
            score[param].append(s)
            
        if i >= max_iter - 1:
            break
            
    ###########################
    ##### Normalize Score #####
    ###########################
    
    for param in dicts:
        score[param] = np.array(score[param]) / normalize_factor[param]
        
    return score


def AUTO_VAE_CIFAR(netE, netG, dicts, max_iter=[1000, 500], loss_type='ELBO', device='cuda:0'):
    
    """ Automated for convenience ! """
    """ loss_type : SHOULD BE 'ELBO' or 'exact' """
    
    opt = config.VAE_cifar10
    Gradients = {}
    
    max_iter1, max_iter2 = max_iter[0], max_iter[1]

    Fisher_inv, normalize_factor, entry_want2see = Calculate_fisher_VAE(
        netE,
        netG,
        TRAIN_loader(
            option=opt.train_dist,
            is_glow=False,
        ),
        dicts,
        opt=opt,
        max_iter=max_iter1,
        loss_type=loss_type,
        noise=0.02,
    )

    for ood in opt.ood_list:
        Gradients[ood] = Calculate_score_VAE(
            netE,
            netG,
            TEST_loader(
                train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            ),
            dicts,
            opt,
            Fisher_inv,
            normalize_factor,
            entry_want2see,
            max_iter=max_iter2,
            loss_type=loss_type,
            noise=0.02,
        )
    
    return Fisher_inv, normalize_factor, Gradients


def AUTO_VAE_FMNIST(netE, netG, dicts, max_iter=[1000, 500], loss_type='ELBO', device='cuda:0'):

    """ Automated for convenience ! """
    """ loss_type : SHOULD BE 'ELBO' or 'exact' """
    
    opt = config.VAE_fmnist
    Gradients = {}
    
    max_iter1, max_iter2 = max_iter[0], max_iter[1]

    Fisher_inv, normalize_factor, entry_want2see = Calculate_fisher_VAE(
        netE,
        netG,
        TRAIN_loader(
            option=opt.train_dist,
            is_glow=False
        ),
        dicts,
        opt=opt,
        max_iter=max_iter1,
        loss_type=loss_type,
        noise=0.02,
    )

    for ood in opt.ood_list:
        Gradients[ood] = Calculate_score_VAE(
            netE,
            netG,
            TEST_loader(
                train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            ),
            dicts,
            opt,
            Fisher_inv,
            normalize_factor,
            entry_want2see,
            max_iter=max_iter2,
            loss_type=loss_type,
            noise=0.02,
        )
    
    return Fisher_inv, normalize_factor, Gradients