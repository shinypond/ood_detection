import numpy as np
from tqdm import tqdm
# If you use jupyter lab, then install @jupyter-widgets/jupyterlab-manager and restart server

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import config
from data_loader import TRAIN_loader, TEST_loader

def Calculate_fisher_CNN(
    model,
    dataloader,
    layers,
    max_iter,
    method='Vanilla',):
    
    """ model : CNN model, pre-trained for FMNIST dataset """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ layers : Which LAYERs do you want to see for calculating Fisher ? """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    assert method == 'SMW' or method == 'Vanilla', 'method must be "SMW" or "Vanilla"'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0) # no learning
    loss_ftn = nn.CrossEntropyLoss(reduction='mean')
    Fisher_inv = {}
    count = 0
    
    for i, (x, y) in enumerate(tqdm(dataloader, desc='Calculate Fisher CNN', unit='step')):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_ftn(y_pred, y)
        loss.backward()
        
        if method == 'SMW':
            assert 0==1
        
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
        assert 0==1
        
    elif method == 'Vanilla':
        normalize_factor = {}
        for lname, _ in layers.items():
            Fisher_inv[lname] = torch.sqrt(Fisher_inv[lname])
            Fisher_inv[lname] = Fisher_inv[lname] * (Fisher_inv[lname] > 1e-3)
            Fisher_inv[lname][Fisher_inv[lname]==0] = 1e-3
            normalize_factor[lname] = 2 * np.sqrt(len(Fisher_inv[lname]))
        
    return Fisher_inv, normalize_factor
        
        

def Calculate_score_CNN(
    model,
    dataloader,
    layers,
    Fisher_inv,
    normalize_factor,
    max_iter,
    method='Vanilla',):
    
    """ model : CNN model, pre-trained for FMNIST dataset """
    """ dataloader : Load 'train distribution' (ex : CIFAR10, FMNIST) """
    """ params : Which PARAMs do you want to see for calculating Fisher ? """
    """ max_iter : When do you want to stop to calculate Fisher ? """
    """ method : 'SMW' (Use Sherman-Morrison-Woodbury Formula) or 'Vanilla' (only see diagonal of Fisher matrix) """
    
    assert method == 'SMW' or method == 'Vanilla', 'method must be "SMW" or "Vanilla"'
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0) # no learning
    loss_ftn = nn.CrossEntropyLoss(reduction='mean')
    score = {}
    temp = 10 # temperature
    
    for i, x in enumerate(tqdm(dataloader, desc='Calculate Fisher CNN', unit='step')):
        
        try: # with label (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # without label (ex : celeba)
            pass
        
        optimizer.zero_grad()
        x = x.to(device)
        y_pred = model(x)
        y = torch.argmax(y_pred, dim=1)
        loss = loss_ftn(y_pred / temp, y)
        loss.backward()
        
        if method == 'SMW':
            assert 0==1
        
        elif method == 'Vanilla':
            grads = {}
            for lname, layer in layers.items():
                grads[lname] = []
                for param in layer.parameters():
                    grads[lname].append(param.grad.view(-1))
                grads[lname] = torch.cat(grads[lname]).to(device)
                s = torch.norm(grads[lname] / Fisher_inv[lname]).detach().cpu()
                if i == 0:
                    score[lname] = []
                score[lname].append(s.numpy())
                
        if i >= max_iter - 1:
            break
    
    for lname, _ in layers.items():
        score[lname] = np.array(score[lname]) / normalize_factor[lname]
        
    return score
            
    
def AUTO_CNN(
    opt,
    model,
    params,
    max_iter=[1000, 500],
    method='Vanilla',
    device='cuda:0'):
    
    """ Automated for convenience ! """
    
    Gradients = {}
    
    Fisher_inv, normalize_factor = Calculate_fisher_CNN(
        model,
        TRAIN_loader(
            option=opt.train_dist,
            is_glow=False,
        ),
        params,
        max_iter=max_iter[0],
        method=method,
    )

    for ood in opt.ood_list:
        Gradients[ood] = Calculate_score_CNN(
            model,
            TEST_loader(
                train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            ),
            params,
            Fisher_inv,
            normalize_factor,
            max_iter=max_iter[1],
            method=method,
        )
    
    return Fisher_inv, normalize_factor, Gradients

    
    
    
    