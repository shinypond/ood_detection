import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
# If you use jupyter lab, then install @jupyter-widgets/jupyterlab-manager and restart server

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import config
from data_loader import TRAIN_loader, TEST_loader

from ekfac_GLOW import EKFACOptimizer

# fix a random seed
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def Calculate_fisher_GLOW_ekfac(
    model,
    opt,
    max_iter,
    select_modules=[],
    seed=2021,):
    
    is_glow = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TRAIN_loader(option=opt.train_dist, is_glow=is_glow, batch_size=1)
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0) # no learning
    ekfac_optim = EKFACOptimizer(model, select_modules=select_modules)
    
    for i, x in enumerate(tqdm(dataloader, desc='Calculate A, B', unit='step')):

        try:
            x, _ = x
        except:
            pass

        optimizer.zero_grad()
        ekfac_optim.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
        ekfac_optim.step()
        
        if i >= max_iter - 1:
            break
            
    A = ekfac_optim.m_aa
    B = ekfac_optim.m_gg
    U_A, U_B = {}, {}
    S = {}
    for name, module in model.named_modules():
        if module in A.keys():
            A[module] += 1e-15 * torch.diag(torch.ones(A[module].shape[0])).to(device)
            B[module] += 1e-15 * torch.diag(torch.ones(B[module].shape[0])).to(device)
            _, U_A[name] = torch.symeig(A[module], eigenvectors=True)
            _, U_B[name] = torch.symeig(B[module], eigenvectors=True)
            S[name] = 0

    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TRAIN_loader(option=opt.train_dist, is_glow=is_glow, batch_size=1)

    for i, x in enumerate(tqdm(dataloader, desc='Calculate Fisher Inverse', unit='step')):
        
        try:
            x, _ = x
        except:
            pass

        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()

        for name, module in model.named_modules():
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
    dataloader = TRAIN_loader(option=opt.train_dist, is_glow=is_glow, batch_size=1)
        
    for i, x in enumerate(tqdm(dataloader, desc=f'Calculate Score of {opt.train_dist}(train)', unit='step')):
        
        try:
            x, _ = x
        except:
            pass
        
        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
        
        for name, module in model.named_modules():
            if module in A.keys():
                GRAD = module.weight.grad.view(module.weight.shape[0], -1) # without bias
                if module.bias is not None:
                    GRAD = torch.cat([GRAD, module.bias.grad.view(module.bias.shape[0], -1)], 1)
                temp = torch.mm(torch.mm(U_B[name].T, GRAD), U_A[name]).view(-1, 1)
                s = temp / (S[name].view(-1, 1) + 1e-15)
                s = torch.mm(temp.T, s).detach().cpu().numpy().reshape(-1)
                
                if name in train_score.keys():
                    train_score[name].append(s)
                else:
                    train_score[name] = []
                    train_score[name].append(s)
                    
        if i >= max_iter - 1:
            break
    
    # Obtain MEAN, STDDEV of ROSE in train-dist at each module.
    mean, std = {}, {}
    for name, module in model.named_modules():
        if module in A.keys():
            mean[name] = np.array(train_score[name]).mean()
            std[name] = np.array(train_score[name]).std()
                    
    return U_A, U_B, S, mean, std, train_score
        
def Calculate_score_GLOW_ekfac(
    model,
    opt,
    U_A,
    U_B,
    S,
    ood,
    max_iter,
    seed=2021):
    
    is_glow = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    random.seed(seed)
    dataloader = TEST_loader(opt.train_dist, ood, is_glow=False)
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0) # no learning
    score = {}
    if ood == opt.train_dist: # i.e, In-dist(test)
        start = datetime.now()
        
    for i, x in enumerate(tqdm(dataloader, desc=f'Calculate Score of {ood}', unit='step')):
        
        try: # with label (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # without label (ex : celeba)
            pass
        
        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
        
        for name, module in model.named_modules():
            if name in U_A.keys():
                GRAD = module.weight.grad.view(module.weight.shape[0], -1) # without bias
                if module.bias is not None:
                    GRAD = torch.cat([GRAD, module.bias.grad.view(module.bias.shape[0], -1)], 1)
                temp = torch.mm(torch.mm(U_B[name].T, GRAD), U_A[name]).view(-1, 1)
                print(name.split('.')[2:4], GRAD.shape, torch.abs(GRAD).max().item(), torch.abs(GRAD).mean().item())
                print(torch.abs(temp).max().item(), torch.abs(temp).mean().item())
                s = temp / (S[name].view(-1, 1) + 1e-15)
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
        
        
        
        
        
        
        
        
def Calculate_fisher_GLOW(
    model,
    dataloader,
    params,
    max_iter,
    method='Vanilla',
    seed=2021):
    
    """ model : trained GLOW model """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ params : Which PARAMs do you want to see for calculating Fisher ? """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0) # no learning
    Fisher_inv = {}
    count = 0
    torch.manual_seed(seed)
    random.seed(seed)

    for i, (x, _) in enumerate(tqdm(dataloader, desc='Calculate Fisher GLOW', unit='step')):

        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
                
        if method == 'SMW':
            grads = {}
            count += 1
            for pname, param in params.items():
                grads[pname] = []
                if np.prod(param.shape) <= 2500: # calculate the exact inverse
                    grads[pname] = param.grad.view(-1).to(device)
                    if i == 0:
                        identity = torch.diag(torch.ones(grads[pname].shape[0]))
                        Fisher_inv[pname] = 1e-3 * identity.to(device)
                    Fisher_inv[pname] += torch.mm(grads[pname].unsqueeze(1), grads[pname].unsqueeze(0))
                else: # calculate the inverse by SMW formula
                    for j in range(param.grad.shape[0]):
                        grads[pname].append(param.grad[j, :, :, :].view(-1, 1))
                    grads[pname] = torch.cat(grads[pname], dim=1).T.to(device)
                    #grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1)
                    if i == 0:
                        identity = torch.diag(torch.ones(grads[pname].shape[1]))
                        Fisher_inv[pname] = 1000 * identity.unsqueeze(0).to(device)
                        Fisher_inv[pname] = Fisher_inv[pname].repeat(grads[pname].shape[0], 1, 1)
                    u1 = grads[pname].unsqueeze(1)
                    u2 = grads[pname].unsqueeze(2)
                    b = torch.bmm(Fisher_inv[pname], u2)
                    denom = torch.ones(grads[pname].shape[0], 1).to(device) + torch.bmm(u1, b).squeeze(2)
                    denom = denom.unsqueeze(2)
                    numer = torch.bmm(b, b.permute(0, 2, 1))
                    Fisher_inv[pname] -= numer / denom
                
        elif method == 'Vanilla':
            grads = {}
            for pname, param in params.items():
                #grads[pname] = []
                #for p in param.parameters():
                #    grads[pname].append(p.grad.view(-1) ** 2)
                #grads[pname] = torch.cat(grads[pname], dim=0)
                #a = np.array((torch.abs(grads[pname]) <= 3e-8).detach().cpu().numpy(), dtype=np.int)
                grads[pname] = param.grad.view(-1) ** 2
                if i == 0:
                    Fisher_inv[pname] = grads[pname]
                else:
                    Fisher_inv[pname] = (i * Fisher_inv[pname] + grads[pname]) / (i + 1)
                    
        if i >= max_iter - 1:
            break
        
    if method == 'SMW':
        normalize_factor = {}
        for pname, param in params.items():
            if np.prod(param.shape) <= 1000: # exact
                Fisher_inv[pname] = count * torch.inverse(Fisher_inv[pname])
            else: # SMW
                Fisher_inv[pname] *= count
            normalize_factor[pname] = 2 * np.sqrt(np.array(Fisher_inv[pname].shape).prod())
            
    elif method == 'Vanilla':
        normalize_factor = {}
        for pname, _ in params.items():
            Fisher_inv[pname] = torch.sqrt(Fisher_inv[pname])
            Fisher_inv[pname] = Fisher_inv[pname] * (Fisher_inv[pname] > 1e-15)
            Fisher_inv[pname][Fisher_inv[pname]==0] = 1e-15
            normalize_factor[pname] = 2 * np.sqrt(len(Fisher_inv[pname]))
        
    ###################3######################3 
    torch.manual_seed(seed)
    random.seed(seed)
    train_score = {}

    for i, (x, _) in enumerate(tqdm(dataloader, desc='Calculate Fisher GLOW', unit='step')):

        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
        
        if method == 'Vanilla':
            grads = {}
            for pname, param in params.items():
                grads[pname] = param.grad.view(-1)
                s = torch.norm(grads[pname] / Fisher_inv[pname]).detach().cpu()
                #s = torch.norm(grads[pname]).detach().cpu()
                if i == 0:
                    train_score[pname] = []
                train_score[pname].append(s.numpy())
                
        if i >= max_iter - 1:
            break
            
    for pname, _ in params.items():
        train_score[pname] = np.array(train_score[pname]) / normalize_factor[pname]
        
    return Fisher_inv, normalize_factor, train_score

def Calculate_score_GLOW(
    model,
    dataloader,
    params,
    Fisher_inv,
    normalize_factor,
    max_iter,
    method='SMW',
    seed=2021):
    
    """ model : trained GLOW model """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ params : Which PARAMs do you want to see for calculating Fisher ? """
    """ Fisher_inv, normalize_factor : Outputs from the function 'Calculate_fisher_GLOW' """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0)
    score = {}
    torch.manual_seed(seed)
    random.seed(seed)
        
    for i, x in enumerate(tqdm(dataloader, desc='Calculate Score GLOW', unit='step')):
        
        try: # with label (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # without label (ex : celeba)
            pass
        
        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
        
        if method == 'SMW_invconv':
            grads = {}
            for pname, param in params.items():
                grads[pname] = []
                
        
        if method == 'SMW':
            grads = {}
            for pname, param in params.items():
                grads[pname] = []
                if np.prod(param.shape) <= 1000: # exact
                    grads[pname] = param.grad.view(-1).to(device)
                    u1 = grads[pname].unsqueeze(0)
                    u2 = grads[pname].unsqueeze(1)
                    s = torch.mm(torch.mm(u1, Fisher_inv[pname]), u2)
                    s = s.view(-1).detach().cpu().numpy()
                else: # SMW
                    for j in range(param.grad.shape[0]):
                        grads[pname].append(param.grad[j, :, :, :].view(-1, 1))
                    grads[pname] = torch.cat(grads[pname], dim=1).T.to(device)
                    grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1)
                    u1 = grads[pname].unsqueeze(1)
                    u2 = grads[pname].unsqueeze(2)
                    s = torch.bmm(torch.bmm(u1, Fisher_inv[pname]), u2)
                    s = torch.sum(s).detach().cpu().numpy()
                if i == 0:
                    score[pname] = []
                score[pname].append(s)
                
        elif method == 'Vanilla':
            grads = {}
            for pname, param in params.items():
                grads[pname] = param.grad.view(-1)
                s = torch.norm(grads[pname] / Fisher_inv[pname]).detach().cpu()
                #s = torch.norm(grads[pname]).detach().cpu()
                if i == 0:
                    score[pname] = []
                score[pname].append(s.numpy())
                
        if i >= max_iter - 1:
            break
            
    for pname, _ in params.items():
        score[pname] = np.array(score[pname]) / normalize_factor[pname]
            
    return score


def AUTO_GLOW(
    opt,
    model,
    params,
    max_iter=[1000, 1000],
    method='SMW',
    device='cuda:0'):
    
    """ Automated for convenience ! """
    
    assert opt.dataroot == '../data', 'please go config.py and modify dataroot to "../../data"'
    
    Gradients = {}
    
    Fisher_inv, normalize_factor, train_score = Calculate_fisher_GLOW(
        model,
        TRAIN_loader(
            option=opt.train_dist,
            is_glow=True,
        ),
        params,
        max_iter=max_iter[0],
        method=method,
    )

    for ood in opt.ood_list:
        Gradients[ood] = Calculate_score_GLOW(
            model,
            TEST_loader(
                train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=True,
            ),
            params,
            Fisher_inv,
            normalize_factor,
            max_iter=max_iter[1],
            method=method,
        )
    
    return Fisher_inv, normalize_factor, train_score, Gradients
    


