import numpy as np
from tqdm import tqdm
# If you use jupyter lab, then install @jupyter-widgets/jupyterlab-manager and restart server

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import config
from data_loader import TRAIN_loader, TEST_loader

def Calculate_fisher_GLOW(
    model,
    dataloader,
    params,
    max_iter,
    method='SMW',):
    
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
                    print(pname, param.shape)
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
                    print(Fisher_inv[pname].shape)
                
        elif method == 'Vanilla':
            grads = {}
            for pname, param in params.items():
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
            Fisher_inv[pname] = Fisher_inv[pname] * (Fisher_inv[pname] > 1e-8)
            Fisher_inv[pname][Fisher_inv[pname]==0] = 1e-8
            normalize_factor[pname] = 2 * np.sqrt(len(Fisher_inv[pname]))
        
    return Fisher_inv, normalize_factor

def Calculate_score_GLOW(
    model,
    dataloader,
    params,
    Fisher_inv,
    normalize_factor,
    max_iter,
    method='SMW',):
    
    """ model : trained GLOW model """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ params : Which PARAMs do you want to see for calculating Fisher ? """
    """ Fisher_inv, normalize_factor : Outputs from the function 'Calculate_fisher_GLOW' """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0)
    score = {}
        
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
    
    Gradients = {}
    
    Fisher_inv, normalize_factor = Calculate_fisher_GLOW(
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
    
    return Fisher_inv, normalize_factor, Gradients
    


