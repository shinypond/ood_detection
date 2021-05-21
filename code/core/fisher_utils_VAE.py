import numpy as np
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
# If you use jupyter lab, then install @jupyter-widgets/jupyterlab-manager and restart server

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from custom_loss import VAE_loss_pixel, loglikelihood
import config
from data_loader import TRAIN_loader, TEST_loader


def Calculate_fisher_VAE(
    netE,
    netG,
    dataloader,
    params,
    opt,
    max_iter,
    loss_type='ELBO_pixel',
    method='SMW',):
    
    """ netE, netG : Encoder, Decoder of trained VAE """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ params : Which PARAMs do you want to see for calculating Fisher ? """
    """ opt : Refer to config.py """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ loss_type : There are two options """
    """     (1) 'ELBO' (traditional VAE loss) """
    """     (2) 'exact' (exact LogLikelihood for VAE ; Jae-moo Choi's NEW(?) viewpoint)"""
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    netE.eval()
    netG.eval()
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    Fisher_inv = {}
    normalize_factor = {}
    grads = {}
    count = 0
    
    for i, (x, _) in enumerate(tqdm(dataloader, desc='Calculate Fisher VAE', unit='step')):
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
        [z, mu, logvar] = netE(x)
        
        if opt.num_samples == 1:
            recon = netG(mu)
        elif opt.num_samples > 1:
            recon = netG(z)
        else:
            raise ValueError
        
        recon = recon.contiguous()
        
        if loss_type == 'ELBO_pixel':
            loss = VAE_loss_pixel(x, [recon, mu, logvar])
        elif loss_type == 'exact':
            loss = loglikelihood(x, z, [recon, mu, logvar])
        else:
            raise ValueError
            
        loss.backward(retain_graph=True)
        
        if method == 'exact':
            count += 1
            for pname, param in params.items():
                grads[pname] = []
                for j in range(param.grad.shape[0]):
                    grads[pname].append(param.grad[j, :, :, :].view(-1, 1))
                grads[pname] = torch.cat(grads[pname], dim=1).T.to(device)
                grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1) # 800 x 1024
                
                if i == 0:
                    identity = torch.diag(torch.ones(grads[pname].shape[1]))
                    Fisher_inv[pname] = 1e-3 * identity.unsqueeze(0).repeat(grads[pname].shape[0], 1, 1).to(device)
                Fisher_inv[pname] += torch.bmm(grads[pname].unsqueeze(2), grads[pname].unsqueeze(1))
                    
        elif method == 'SMW':
            count += 1
            for pname, param in params.items():
                grads[pname] = []
                for j in range(param.grad.shape[0]):
                    grads[pname].append(param.grad[j, :, :, :].view(-1, 1))
                grads[pname] = torch.cat(grads[pname], dim=1).T.to(device)
                grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1)
                
                if i == 0:
                    Fisher_inv[pname] = 1000 * torch.diag(torch.ones(grads[pname].shape[1])).unsqueeze(0).to(device)
                    Fisher_inv[pname] = Fisher_inv[pname].repeat(grads[pname].shape[0], 1, 1)
                    
                u1 = grads[pname].unsqueeze(1)
                u2 = grads[pname].unsqueeze(2)
                b = torch.bmm(Fisher_inv[pname], u2)
                denom = torch.ones(grads[pname].shape[0], 1).to(device) + torch.bmm(u1, b).squeeze(2)
                denom = denom.unsqueeze(2)
                numer = torch.bmm(b, b.permute(0, 2, 1))
                Fisher_inv[pname] -= numer / denom
                
        elif method == 'Vanilla':
            for pname, param in params.items():
                grads[pname] = param.grad.view(-1) ** 2
                if i == 0:
                    Fisher_inv[pname] = grads[pname]
                else:
                    Fisher_inv[pname] = (i * Fisher_inv[pname] + grads[pname]) / (i + 1)
                    
        if i >= max_iter - 1:
            break
    
    if method == 'exact':
        for pname, _ in params.items():
            for j in range(Fisher_inv[pname].shape[0]):
                Fisher_inv[pname][j, :, :] = count * torch.inverse(Fisher_inv[pname][j, :, :])
                normalize_factor[pname] = 2 * np.sqrt(np.array(Fisher_inv[pname].shape).prod())
    
    elif method == 'SMW':
        for pname, _ in params.items():
            Fisher_inv[pname] *= count
            normalize_factor[pname] = 2 * np.sqrt(np.array(Fisher_inv[pname].shape).prod())
            
    elif method == 'Vanilla':
        for pname, _ in params.items():
            Fisher_inv[pname] = torch.sqrt(Fisher_inv[pname])
            Fisher_inv[pname] = Fisher_inv[pname] * (Fisher_inv[pname] > 1e-3)
            Fisher_inv[pname][Fisher_inv[pname]==0] = 1e-3
            normalize_factor[pname] = 2 * np.sqrt(len(Fisher_inv[pname]))
        
    return Fisher_inv, normalize_factor


def Calculate_score_VAE(
    netE,
    netG,
    dataloader,
    params,
    opt,
    Fisher_inv,
    normalize_factor,
    max_iter,
    loss_type='ELBO_pixel',
    method='SMW',):
    
    """ netE, netG : Encoder, Decoder of trained VAE """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ params : Which LAYERs do you want to see for calculating Fisher ? """
    """ opt : Refer to config.py """
    """ Fisher_inv, normalize_factor : Outputs from the function 'Calculate_fisher_VAE' """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ loss_type : There are two options """
    """     (1) 'ELBO' (traditional VAE loss) """
    """     (2) 'exact' (exact LogLikelihood for VAE ; Jae-moo Choi's NEW(?) viewpoint)"""
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    netE.eval()
    netG.eval()
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    grads = {}
    score = {}
    start = datetime.now()
        
    for i, x in enumerate(tqdm(dataloader, desc='Calculate Score VAE', unit='step')):
        
        try: # with label (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # without label (ex : celeba)
            pass
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
        gradient_val = 0
        [z, mu, logvar] = netE(x)
        
        if opt.num_samples == 1:
            recon = netG(mu)
        elif opt.num_samples > 1:
            recon = netG(z)
        else:
            raise ValueError
            
        recon = recon.contiguous()
        
        if loss_type == 'ELBO_pixel':
            loss = VAE_loss_pixel(x, [recon, mu, logvar])
        elif loss_type == 'exact':
            loss = loglikelihood(x, z, [recon, mu, logvar])
        else:
            raise ValueError
            
        loss.backward(retain_graph=True)
        
        if method == 'SMW' or method == 'exact':
            for pname, param in params.items():
                grads[pname] = []
                for j in range(param.grad.shape[0]):
                    grads[pname].append(param.grad[j, :, :, :].view(-1, 1)) # 4096 x 1
                grads[pname] = torch.cat(grads[pname], dim=1).T.to(device) # 200 x 4096
                grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1) # (200 x 64) x (4096 / 64)
                u1 = grads[pname].unsqueeze(1)
                u2 = grads[pname].unsqueeze(2)
                s = torch.bmm(torch.bmm(u1, Fisher_inv[pname]), u2)
                s = torch.sum(s).detach().cpu().numpy()
                if i == 0:
                    score[pname] = []
                score[pname].append(s)
            end = datetime.now()
                
        elif method == 'Vanilla':
            for pname, param in params.items():
                grads[pname] = param.grad.view(-1)
                s = torch.norm(grads[pname] / Fisher_inv[pname]).detach().cpu()
                if i == 0:
                    score[pname] = []
                score[pname].append(s.numpy())
                
        if i >= max_iter - 1:
            break
    end = datetime.now()
    avg_time = (end - start) / max_iter
            
    for pname, _ in params.items():
        score[pname] = np.array(score[pname]) / normalize_factor[pname]
            
    return score, avg_time


def AUTO_VAE(
    opt,
    netE,
    netG,
    params,
    max_iter=[1000, 1000],
    loss_type='ELBO_pixel',
    method='SMW',
    device='cuda:0'):
    
    """ Automated for convenience ! """
    """ loss_type : SHOULD BE 'ELBO_pixel' or 'exact' """
    
    assert opt.dataroot == '../data', 'please go config.py and modify dataroot to "../../data"'
    
    Gradients = {}
    
    Fisher_inv, normalize_factor = Calculate_fisher_VAE(
        netE,
        netG,
        TRAIN_loader(
            option=opt.train_dist,
            is_glow=False,
        ),
        params,
        opt=opt,
        max_iter=max_iter[0],
        loss_type=loss_type,
        method=method,
    )

    for ood in opt.ood_list:
        Gradients[ood], avg_time = Calculate_score_VAE(
            netE,
            netG,
            TEST_loader(
                train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            ),
            params,
            opt,
            Fisher_inv,
            normalize_factor,
            max_iter=max_iter[1],
            loss_type=loss_type,
            method=method,
        )
        if ood == opt.train_dist:
            TIME = avg_time
    
    return Fisher_inv, normalize_factor, Gradients, TIME




# TRASH CAN (mu.grad) - 21.05.08
"""
#########################
##### Use mu_grad ! #####
#########################
    grad = torch.mean(mu.grad, dim=0, keepdim=False).squeeze(-1)
    count += 1
    if i == 0:
        Fisher_inv['mu'] = 1000 * torch.diag(torch.ones(grad.shape[0])).to(device)
    b = torch.mm(Fisher_inv['mu'], grad)
    denom = 1 + torch.mm(grad.T, b)    
    numer = torch.mm(b, b.T)
    Fisher_inv['mu'] -= numer / denom
    if i >= max_iter - 1:
        break
normalize_factor = {}
Fisher_inv['mu'] *= count
normalize_factor['mu'] = 2 * np.sqrt(np.array(Fisher_inv['mu'].shape).prod())
"""

"""
#########################
##### Use mu.grad ! #####
#########################
    grad = torch.mean(mu.grad, dim=0, keepdim=False).squeeze(-1)
    s = torch.mm(torch.mm(grad.T, Fisher_inv['mu']), grad).squeeze(-1).squeeze(-1)
    if i == 0:
        score['mu'] = []
    score['mu'].append(s.detach().cpu().numpy())
    if i >= max_iter - 1:
        break
score['mu'] = np.array(score['mu']) / normalize_factor['mu']
"""