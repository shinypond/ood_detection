import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
# If you use jupyter lab, then install @jupyter-widgets/jupyterlab-manager and restart server

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from custom_loss import VAE_loss_pixel, loglikelihood
import config
from data_loader import TRAIN_loader, TEST_loader
from visualize import AUROC

from ekfac_VAE import EKFACOptimizer

# fix a random seed
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def Calculate_fisher_VAE_ekfac(
    netE,
    netG,
    opt,
    max_iter,
    select_modules=[],
    seed=2021,):
    
    is_glow = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TRAIN_loader(option=opt.train_dist, shuffle=True, is_glow=is_glow, batch_size=1)
    netE.eval()
    netG.eval()
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    ekfac_optim = EKFACOptimizer(netE, select_modules=select_modules)
    
    for i, x in enumerate(tqdm(dataloader, desc='Calculate A, B', unit='step')):
        
        try:
            x, _ = x
        except:
            pass
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        ekfac_optim.zero_grad()
        x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
        [z, mu, logvar] = netE(x)
        
        if opt.num_samples == 1:
            recon = netG(mu)
        elif opt.num_samples > 1:
            recon = netG(z)
        else:
            raise ValueError
        
        recon = recon.contiguous()
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
        ekfac_optim.step()
        
        if i >= max_iter - 1:
            break
        
    A = ekfac_optim.m_aa
    B = ekfac_optim.m_gg
    U_A, U_B = {}, {}
    S = {}
    for name, module in netE.named_modules():
        if module in A.keys():
            A[module] += 1e-12 * torch.diag(torch.ones(A[module].shape[0])).to(device)
            B[module] += 1e-12 * torch.diag(torch.ones(B[module].shape[0])).to(device)
            _, U_A[name] = torch.symeig(A[module], eigenvectors=True)
            _, U_B[name] = torch.symeig(B[module], eigenvectors=True)
            S[name] = 0
    
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TRAIN_loader(option=opt.train_dist, shuffle=True, is_glow=is_glow, batch_size=1)
    
    for i, x in enumerate(tqdm(dataloader, desc='Calculate Fisher Inverse', unit='step')):
        
        try:
            x, _ = x
        except:
            pass
        
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
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
        
        for name, module in netE.named_modules():
            if module in A.keys():
                GRAD = module.weight.grad.view(module.weight.shape[0], -1) # without bias
                if module.bias is not None:
                    GRAD = torch.cat([GRAD, module.bias.grad.view(module.bias.shape[0], -1)], 1)
                s = torch.mm(torch.mm(U_B[name].T, GRAD), U_A[name]).view(-1)
                s = s ** 2
                S[name] = (i * S[name] + s.clone().detach()) / (i + 1)
        
        if i >= max_iter - 1:
            break
    
    train_score = {}
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TRAIN_loader(option=opt.train_dist, shuffle=True, is_glow=is_glow, batch_size=1)
    
    for i, x in enumerate(tqdm(dataloader, desc=f'Calculate Score of {opt.train_dist}(train)', unit='step')):
        
        try:
            x, _ = x
        except:
            pass
        
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
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
        
        for name, module in netE.named_modules():
            if module in A.keys():
                GRAD = module.weight.grad.view(module.weight.shape[0], -1) # without bias
                if module.bias is not None:
                    GRAD = torch.cat([GRAD, module.bias.grad.view(module.bias.shape[0], -1)], 1)
                temp = torch.mm(torch.mm(U_B[name].T, GRAD), U_A[name]).view(-1, 1)
                s = temp / (S[name].view(-1, 1) + 1e-8)
                s = torch.mm(temp.T, s).detach().cpu().numpy().reshape(-1)
                
                if name in train_score.keys():
                    train_score[name].append(s)
                else:
                    train_score[name] = []
                    train_score[name].append(s)
                    
        if i >= max_iter - 1:
            break
    
    # Obtain MEAN, STDDEV of ROSE in train-dist at each module in the encoder (netE).
    mean, std = {}, {}
    for name, module in netE.named_modules():
        if module in A.keys():
            mean[name] = np.array(train_score[name]).mean()
            std[name] = np.array(train_score[name]).std()
                    
    return U_A, U_B, S, mean, std

def Calculate_score_VAE_ekfac(
    netE,
    netG,
    opt,
    U_A,
    U_B,
    S,
    ood,
    max_iter,
    seed=2021):
    
    is_glow = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TEST_loader(opt.train_dist, ood, shuffle=True, is_glow=is_glow)
    netE.eval()
    netG.eval()
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    score = {}
    if ood == opt.train_dist: # i.e, In-dist(test)
        start = datetime.now()
        
    for i, x in enumerate(tqdm(dataloader, desc=f'Calculate Score of {ood}', unit='step')):
        
        try: # with label (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # without label (ex : celeba)
            pass
        
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
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
        
        for name, module in netE.named_modules():
            if name in U_A.keys():
                GRAD = module.weight.grad.view(module.weight.shape[0], -1) # without bias
                if module.bias is not None:
                    GRAD = torch.cat([GRAD, module.bias.grad.view(module.bias.shape[0], -1)], 1)
                temp = torch.mm(torch.mm(U_B[name].T, GRAD), U_A[name]).view(-1, 1)
                s = temp / (S[name].view(-1, 1) + 1e-8)
                s = torch.mm(temp.T, s).detach().cpu().numpy().reshape(-1)
                
                if name in score.keys():
                    score[name].append(s)
                else:
                    score[name] = []
                    score[name].append(s)
        
        if i >= max_iter - 1:
            break
            
    for name in score.keys():
        score[name] = np.array(score[name])
    
    if ood == opt.train_dist:
        end = datetime.now()
        avg_time = (end - start).total_seconds() / max_iter
        print(f'Average Inference Time : {avg_time} seconds')
        print(f'Average #Images Processed : {1 / avg_time} Images')
    
    return score
        
        

        
        
        
        
        
        


def Calculate_fisher_VAE_diag(
    netE,
    netG,
    params,
    opt,
    max_iter,):
    
    is_glow = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TRAIN_loader(opt.train_dist, is_glow=is_glow)
    netE.eval()
    netG.eval()
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    Fisher_inv = {}
    normalize_factor = {}
    grads = {}
    count = 0
    
    for i, (x, _) in enumerate(tqdm(dataloader, desc=f'Calculate Fisher Inverse', unit='step')):
        
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
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
        
        for pname, param in params.items():
            grads[pname] = param.grad.view(-1) ** 2
            if i == 0:
                Fisher_inv[pname] = grads[pname]
            else:
                Fisher_inv[pname] = (i * Fisher_inv[pname] + grads[pname]) / (i + 1)
                    
        if i >= max_iter - 1:
            break
    
    for pname, _ in params.items():
        Fisher_inv[pname] = torch.sqrt(Fisher_inv[pname])
        Fisher_inv[pname] = Fisher_inv[pname] * (Fisher_inv[pname] > 1e-8)
        Fisher_inv[pname][Fisher_inv[pname]==0] = 1e-8
        normalize_factor[pname] = 2 * np.sqrt(len(Fisher_inv[pname]))
        
    train_score = {}
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TRAIN_loader(opt.train_dist, is_glow=is_glow)
    
    for i, (x, _) in enumerate(tqdm(dataloader, desc=f'Calculate Score of {opt.train_dist}(train)', unit='step')):
        
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
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
        
        for pname, param in params.items():
            grads[pname] = param.grad.view(-1)
            s = torch.norm(grads[pname] / Fisher_inv[pname]).detach().cpu().numpy().reshape(-1)
            if i == 0:
                train_score[pname] = []
            train_score[pname].append(s)
            
        if i >= max_iter - 1:
            break
            
    # Obtain MEAN, STDDEV of ROSE in train-dist at each module in the encoder (netE).
    mean, std = {}, {}
    for pname, param in params.items():
        mean[pname] = np.array(train_score[pname]).mean()
        std[pname] = np.array(train_score[pname]).std()
    
    return Fisher_inv, normalize_factor, mean, std


def Calculate_score_VAE_diag(
    netE,
    netG,
    params,
    opt,
    Fisher_inv,
    normalize_factor,
    max_iter,
    ood,):
    
    is_glow = False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TEST_loader(opt.train_dist, ood, shuffle=True, is_glow=is_glow)
    netE.eval()
    netG.eval()
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
    grads = {}
    score = {}
        
    for i, x in enumerate(tqdm(dataloader, desc=f'Calculate Score of {ood}', unit='step')):
        
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
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
            
        loss.backward(retain_graph=True)
                
        for pname, param in params.items():
            grads[pname] = param.grad.view(-1)
            s = torch.norm(grads[pname] / Fisher_inv[pname]).detach().cpu().numpy().reshape(-1)
            if i == 0:
                score[pname] = []
            score[pname].append(s)
                
        if i >= max_iter - 1:
            break
            
    for pname, _ in params.items():
        score[pname] = np.array(score[pname]) / normalize_factor[pname]
            
    return score

def AUTO_VAE(
    opt,
    netE,
    netG,
    select_modules,
    max_iter=[1000, 1000],
    method='ekfac(max)'):
    
    assert opt.dataroot == '../data', 'please go config.py and modify dataroot to "../../data"'
    
    auroc = {}
    SCOREs = {}
    
    if method == 'ekfac(max)':
        U_A, U_B, S, mean, std = Calculate_fisher_VAE_ekfac(netE, netG, opt, select_modules=modules, max_iter=max_iter[0])
        
        for ood in opt.ood_list:
            SCOREs[ood] = Calculate_score_VAE_ekfac(netE, netG, opt, U_A, U_B, S, ood, max_iter=max_iter[1])
        
    elif method == 'diag':
        Fisher_inv, normalize_factor, mean, std = Calculate_fisher_VAE_diag(netE, netG, select_modules, opt=opt, max_iter=max_iter[0])

        for ood in opt.ood_list:
            SCOREs[ood] = Calculate_score_VAE_diag(netE, netG, select_modules, opt, Fisher_inv, normalize_factor, max_iter=max_iter[1], ood=ood)
            
    else:
        raise ValueError
        
    print(f'< RESULT > VAE_{method}')
    for ood in opt.ood_list:
        temp = []
        for name in SCOREs[ood].keys():
            a = np.array(SCOREs[ood][name])
            a = (a - mean[name]) / std[name]
            temp.append(a)
        SCOREs[ood] = np.max(np.concatenate(temp, 1), 1)
        args = [SCOREs[opt.train_dist], SCOREs[ood]]
        labels = [opt.train_dist, ood]
        auroc[ood] = AUROC(*args, labels=labels, verbose=False)
        print(f'{opt.train_dist}/{ood} {auroc[ood]}')
    
    return SCOREs, auroc

