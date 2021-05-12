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
    layers,
    max_iter,
    method='SMW',):
    
    """ model : trained GLOW model """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ layers : Which LAYERs do you want to see for calculating Fisher ? """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    assert method == 'SMW' or method == 'Vanilla', 'method must be "SMW" or "Vanilla"'
    
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
            assert 0==1, 'Have not been developed yet,,,'
            grads = {}
            count += 1
            for lname, layer in layers.items():
                grads[lname] = []
                for param in layer.parameters():
                    grads[lname].append(param.grad.view(-1, 1))
                grads[lname] = torch.cat(grads[lname]).to(device)
                grads[lname] = grads[lname].reshape(grads[lname].shape[0] * 1, -1)
                
                if i == 0:
                    Fisher_inv[lname] = 100 * torch.diag(torch.ones(grads[lname].shape[1]))
                
                print(grads[lname].shape)
                    
                
                
                
                
                
                grads[pname] = []
                for j in range(param.grad.shape[0]):
                    grads[pname].append(param.grad[j, :, :, :].view(-1, 1))
                grads[pname] = torch.cat(grads[pname], dim=1).T.to(device)
                grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1)
                
                if i == 0:
                    Fisher_inv[pname] = 100 * torch.diag(torch.ones(grads[pname].shape[1])).unsqueeze(0).to(device)
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
            for lname, layer in layers.items():
                grads[lname] = []
                for param in layer.parameters():
                    grads[lname].append(param.grad.view(-1) ** 2)
                grads[lname] = torch.cat(grads[lname]).to(device)
                if i == 0:
                    Fisher_inv[lname] = grads[lname]
                else:
                    Fisher_inv[lname] = (i * Fisher_inv[lname] + grads[lname]) / (i + 1)
                    
        if i >= max_iter - 1:
            break
        
    if method == 'SMW':
        normalize_factor = {}
        for pname, _ in params.items():
            Fisher_inv[pname] *= count
            normalize_factor[pname] = 2 * np.sqrt(np.array(Fisher_inv[pname].shape).prod())
            
    elif method == 'Vanilla':
        normalize_factor = {}
        for lname, _ in layers.items():
            Fisher_inv[lname] = torch.sqrt(Fisher_inv[lname])
            Fisher_inv[lname] = Fisher_inv[lname] * (Fisher_inv[lname] > 1e-3)
            Fisher_inv[lname][Fisher_inv[lname]==0] = 1e-3
            normalize_factor[lname] = 2 * np.sqrt(len(Fisher_inv[lname]))
        
    return Fisher_inv, normalize_factor

def Calculate_score_GLOW(
    model,
    dataloader,
    layers,
    Fisher_inv,
    normalize_factor,
    max_iter,
    method='SMW',):
    
    """ model : trained GLOW model """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ layers : Which LAYERs do you want to see for calculating Fisher ? """
    """ Fisher_inv, normalize_factor : Outputs from the function 'Calculate_fisher_GLOW' """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    assert method == 'SMW' or method == 'Vanilla', 'method must be "SMW" or "Vanilla"'
    
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
        
        if method == 'SMW':
            assert 0==1, 'Have not been developed yet,,,'
            grads = {}
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
                
        elif method == 'Vanilla':
            grads = {}
            for lname, layer in layers.items():
                grads[lname] = []
                for param in layer.parameters(): 
                    grads[lname].append(param.grad.view(-1))
                grads[lname] = torch.cat(grads[lname])
                s = torch.norm(grads[lname] / Fisher_inv[lname]).detach().cpu()
                if i == 0:
                    score[lname] = []
                score[lname].append(s.numpy())
                
        if i >= max_iter - 1:
            break
            
    for lname, _ in layers.items():
        score[lname] = np.array(score[lname]) / normalize_factor[lname]
            
    return score


def AUTO_GLOW(
    opt,
    model,
    layers,
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
        layers,
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
            layers,
            Fisher_inv,
            normalize_factor,
            max_iter=max_iter[1],
            method=method,
        )
    
    return Fisher_inv, normalize_factor, Gradients
    


