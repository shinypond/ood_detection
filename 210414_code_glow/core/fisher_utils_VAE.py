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

def Calculate_score_VAE(netE,
                        netG,
                        dataloader,
                        dicts,
                        Grads,
                        normalize_factor,
                        max_iter,
                        num_samples,
                        with_label=True,
                        device='cuda:0'):
    
    """ with_label : If len(dataset[0]) == 2, TRUE. Otherwise, FALSE """
    
    optimizer1 = optim.SGD(netE.parameters(), lr=0,momentum=0)
    optimizer2 = optim.SGD(netG.parameters(), lr=0,momentum=0)
    CE_loss = nn.CrossEntropyLoss(reduction = 'none')
    score = []
        
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
        kld = KL_div(mu, logvar)
        recon = netG(z)
        recon = recon.contiguous()
        recon = recon.view(-1, 256)
        recl = CE_loss(recon, target)
        loss = torch.sum(recl) / b + kld.mean()
        loss.backward(retain_graph=True)

        grads = []
        for param in dicts:
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        gradient_val = torch.norm(grads / Grads) / normalize_factor
        score.append(gradient_val.detach().cpu())
        if i > max_iter:
            break
                
    return score

def AUTO_VAE_CIFAR(netE, netG, dicts, device='cuda:0'):
    
    """ 편하게 자동화한 것입니다. """
    """ max_iter 설정은 여기 함수 내부에서 해 주세요~ """
    
    opt = config.VAE_cifar10

    Grads, normalize_factor = Calculate_fisher_VAE(netE,
                                                   netG,
                                                   TRAIN_loader(option='cifar10',
                                                                is_glow=False),
                                                   dicts,
                                                   max_iter=10000,
                                                   num_samples=opt.num_samples,
                                                  )

    cifar_Gradients = Calculate_score_VAE(netE,
                                          netG,
                                          TEST_loader(train_dist='cifar10',
                                                      target_dist='cifar10',
                                                      is_glow=False),
                                          dicts,
                                          Grads,
                                          normalize_factor,
                                          max_iter=5000,
                                          num_samples=opt.num_samples,
                                          with_label=opt.with_label,
                                         )

    svhn_Gradients = Calculate_score_VAE(netE,
                                         netG,
                                         TEST_loader(train_dist='cifar10',
                                                     target_dist='svhn',
                                                     is_glow=False),
                                         dicts,
                                         Grads,
                                         normalize_factor,
                                         max_iter=5000,
                                         num_samples=opt.num_samples,
                                         with_label=opt.with_label,
                                        )

    celeba_Gradients = Calculate_score_VAE(netE,
                                           netG,
                                           TEST_loader(train_dist='cifar10',
                                                       target_dist='celeba',
                                                       is_glow=False),
                                           dicts,
                                           Grads,
                                           normalize_factor,
                                           max_iter=5000,
                                           num_samples=opt.num_samples,
                                           with_label=opt.with_label,
                                          )

    lsun_Gradients = Calculate_score_VAE(netE,
                                         netG,
                                         TEST_loader(train_dist='cifar10',
                                                     target_dist='lsun',
                                                     shuffle=True,
                                                     is_glow=False),
                                         dicts,
                                         Grads,
                                         normalize_factor,
                                         max_iter=5000,
                                         num_samples=opt.num_samples,
                                         with_label=opt.with_label,
                                        )

    noise_Gradients = Calculate_score_VAE(netE,
                                          netG,
                                          TEST_loader(train_dist='cifar10',
                                                      target_dist='noise',
                                                      is_glow=False),
                                          dicts,
                                          Grads,
                                          normalize_factor,
                                          max_iter=5000,
                                          num_samples=opt.num_samples,
                                          with_label=opt.with_label,
                                         )
    
    return Grads, normalize_factor, cifar_Gradients, svhn_Gradients, celeba_Gradients, lsun_Gradients, noise_Gradients


def AUTO_VAE_FMNIST(netE, netG, dicts, device='cuda:0'):

    """ 편하게 자동화한 것입니다. """
    """ max_iter 설정은 여기 함수 내부에서 해 주세요~ """
    
    opt = config.VAE_fmnist

    Grads, normalize_factor = Calculate_fisher_VAE(netE,
                                                   netG,
                                                   TRAIN_loader(option='fmnist',
                                                                is_glow=False),
                                                   dicts,
                                                   max_iter=10000,
                                                   num_samples=opt.num_samples,
                                                  )

    fmnist_Gradients = Calculate_score_VAE(netE,
                                           netG,
                                           TEST_loader(train_dist='fmnist',
                                                       target_dist='fmnist',
                                                       is_glow=False),
                                           dicts,
                                           Grads,
                                           normalize_factor,
                                           max_iter=5000,
                                           num_samples=opt.num_samples,
                                           with_label=opt.with_label,
                                          )

    mnist_Gradients = Calculate_score_VAE(netE,
                                          netG,
                                          TEST_loader(train_dist='fmnist',
                                                      target_dist='mnist',
                                                      is_glow=False),
                                          dicts,
                                          Grads,
                                          normalize_factor,
                                          max_iter=5000,
                                          num_samples=opt.num_samples,
                                          with_label=opt.with_label,
                                         )

    noise_Gradients = Calculate_score_VAE(netE,
                                          netG,
                                          TEST_loader(train_dist='fmnist',
                                                      target_dist='noise',
                                                      is_glow=False),
                                          dicts,
                                          Grads,
                                          normalize_factor,
                                          max_iter=5000,
                                          num_samples=opt.num_samples,
                                          with_label=opt.with_label,
                                         )
    
    return Grads, normalize_factor, fmnist_Gradients, mnist_Gradients, noise_Gradients