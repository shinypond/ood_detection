import numpy as np
from tqdm import tqdm
# If you use jupyter lab, then install @jupyter-widgets/jupyterlab-manager and restart server

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from custom_loss import KL_div, VAE_loss
import config
from data_loader import TRAIN_loader, TEST_loader

def Calculate_fisher_VAE(netE,
                         netG,
                         dataloader,
                         dicts,
                         max_iter,
                         num_samples,
                         device='cuda:0'):

    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0)
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0)
    CE_loss = nn.CrossEntropyLoss(reduction='none')
    Fisher_inv = {}
    count = 0
    
    for i, (x, _) in enumerate(tqdm(dataloader, desc='Calculate Fisher VAE', unit='step')):
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x = x.repeat(num_samples, 1, 1, 1).to(device)
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        [z, mu, logvar] = netE(x)
        kld = KL_div(mu, logvar)
        recon = netG(z)
        recon = recon.contiguous()
        recon = recon.view(-1, 256)
        recl = CE_loss(recon, target)
        loss = torch.sum(recl) / b + kld.mean()
        loss.backward(retain_graph=True)
        
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
        
    
    normalize_factor = {}
    for param in dicts:
        Fisher_inv[param] *= count
        
        # just try....
        entry_want2see = (torch.sum((torch.abs(Fisher_inv[param]) >= 0), dim=[1, 2]) != 0) # 3200
        Fisher_inv[param] = Fisher_inv[param][entry_want2see, :, :]
        
        normalize_factor[param] = 2 * np.sqrt(np.array(Fisher_inv[param].shape).prod())
                
    return Fisher_inv, normalize_factor, entry_want2see
                
                
        
    """ # previous version (Not Sherman-Morrison)
    
        grads = []
        for param in dicts:
            grads.append(param.grad.view(-1) ** 2) 
        grads = torch.cat(grads)
        
        if i == 0:
            Grads = grads
        else:
            Grads = (i * Grads + grads) / (i + 1)
            if i > max_iter:
                break

    Grads = torch.sqrt(Grads)
    Grads = Grads * (Grads > 1e-3)
    Grads[Grads == 0] = 1e-3
    normalize_factor = 2 * np.sqrt(len(Grads))
    
    return Grads, normalize_factor
    """


def Calculate_score_VAE(netE,
                        netG,
                        dataloader,
                        dicts,
                        #Grads,
                        Fisher_inv,
                        normalize_factor,
                        entry_want2see,
                        max_iter,
                        num_samples,
                        with_label=True,
                        device='cuda:0'):
    
    """ with_label : If len(dataset[0]) == 2, TRUE. Otherwise, FALSE """
    
    optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0)
    optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0)
    CE_loss = nn.CrossEntropyLoss(reduction = 'none')
    score = {}
    losses = []
        
    for i, x in enumerate(tqdm(dataloader, desc='Calculate Score VAE', unit='step')):
        
        try: # with_label == True (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # with_label == False (ex : celeba)
            pass
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        x = x.repeat(num_samples, 1, 1, 1).to(device)
        b = x.size(0)
        target = Variable(x.data.view(-1) * 255).long()
        gradient_val = 0
        [z, mu, logvar] = netE(x)
        # z ~~~ mu (interpolation)
        # For each z*, get recl and do backprop
        # output : each grad? relation?
        
        kld = KL_div(mu, logvar)
        recon = netG(z)
        recon = recon.contiguous()
        recon = recon.view(-1, 256)
        recl = CE_loss(recon, target)
        loss = torch.sum(recl) / b + kld.mean()
        loss.backward(retain_graph=True)

        grads = {}
        for param in dicts:
            grads[param] = []
            for j in range(param.grad.shape[0]):
                grads[param].append(param.grad[j, :, :, :].view(-1, 1)) # 1 x 4096
            grads[param] = torch.cat(grads[param], dim=1).T.to(device) # 200 x 4096
            # try
            grads[param] = grads[param].reshape(grads[param].shape[0] * 16, -1) # 3200 x 256
            
            # try
            grads[param] = grads[param][entry_want2see, :] # ???? x 256
            
            u1 = grads[param].unsqueeze(1) # 3200 x 1 x 256
            u2 = grads[param].unsqueeze(2) # 3200 x 256 x 1
            
            s = torch.bmm(torch.bmm(u1, Fisher_inv[param]), u2) # 3200 x 1 x 1
            s = s.squeeze(1).squeeze(1) # 3200
            #n = torch.bmm(u1, u2).squeeze(1).squeeze(1) # 3200
            #s = s / (n + 1e-8)
            s = torch.sum(s).detach().cpu().numpy()
            if i == 0:
                score[param] = []
            score[param].append(s)
        if i >= max_iter - 1:
            break
    for param in dicts:
        score[param] = np.array(score[param]) / normalize_factor[param]
        
    return score
        
    """ # previous version (Not Sherman-Morrison)
        grads = []
        for param in dicts:
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        gradient_val = torch.norm(grads / Grads) / normalize_factor
        losses.append(loss.detach().cpu().numpy())
        score.append(gradient_val.detach().cpu())
        if i > max_iter:
            break
    score = np.array(score)
    losses = np.array(losses)
                
    return score, losses
    """

def AUTO_VAE_CIFAR(netE, netG, dicts, device='cuda:0'):
    
    """ 편하게 자동화한 것입니다. """
    """ max_iter 설정은 여기 함수 내부에서 해 주세요~ """
    
    opt = config.VAE_cifar10
    Gradients = {}
    
    max_iter1 = 1000
    max_iter2 = 500

    Fisher_inv, normalize_factor, entry_want2see = Calculate_fisher_VAE(
        netE,
        netG,
        TRAIN_loader(
            option=opt.train_dist,
            is_glow=False,
        ),
        dicts,
        max_iter=max_iter1,
        num_samples=opt.num_samples,
    )

    for ood in opt.ood_list:
        Gradients[ood] = Calculate_score_VAE(
            netE,
            netG,
            TEST_loader(
                opt.train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            ),
            dicts,
            Fisher_inv,
            normalize_factor,
            entry_want2see,
            max_iter=max_iter2,
            num_samples=opt.num_samples,
            with_label=opt.with_label,
        )
    
    return Fisher_inv, normalize_factor, Gradients


def AUTO_VAE_FMNIST(netE, netG, dicts, device='cuda:0'):

    """ 편하게 자동화한 것입니다. """
    """ max_iter 설정은 여기 함수 내부에서 해 주세요~ """
    
    opt = config.VAE_fmnist
    Gradients = {}
    
    max_iter1 = 1000
    max_iter2 = 500

    Fisher_inv, normalize_factor, entry_want2see = Calculate_fisher_VAE(
        netE,
        netG,
        TRAIN_loader(
            option=opt.train_dist,
            is_glow=False
        ),
        dicts,
        max_iter=max_iter1,
        num_samples=opt.num_samples,
    )

    for ood in opt.ood_list:
        Gradients[ood] = Calculate_score_VAE(
            netE,
            netG,
            TEST_loader(
                opt.train_dist=opt.train_dist,
                target_dist=ood,
                shuffle=True,
                is_glow=False,
            ),
            dicts,
            Fisher_inv,
            normalize_factor,
            entry_want2see,
            max_iter=max_iter2,
            num_samples=opt.num_samples,
            with_label=opt.with_label,
        )
    
    return Fisher_inv, normalize_factor, Gradients