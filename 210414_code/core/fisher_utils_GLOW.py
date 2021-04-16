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

def Calculate_fisher_GLOW(model,
                          dataloader,
                          dicts,
                          max_iter,
                          device='cuda:0'):
    
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0)

    for i, (x, _) in enumerate(tqdm(dataloader, desc='Calculate Fisher GLOW', unit='step')):

        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
        
        grads = []
        for w in dicts:
            for param in w.parameters():
                grads.append(param.grad.view(-1) ** 2)
        grads = torch.cat(grads)
        
        if i == 0:
            Grads = grads
        else:
            Grads = (i * Grads + grads) / (i+1)
            if i > max_iter:
                break

    Grads = torch.sqrt(Grads)
    Grads = Grads * (Grads > 1e-8)
    Grads[Grads == 0] = 1e-8
    normalize_factor = 2 * np.sqrt(len(Grads))
    
    return Grads, normalize_factor
    
def Calculate_score_GLOW(model,
                         dataloader,
                         dicts,
                         Grads,
                         normalize_factor,
                         max_iter,
                         with_label=True,
                         device='cuda:0'):
    
    """ with_label : If len(dataset[0]) == 2, TRUE. Otherwise, FALSE """
    
    optimizer = optim.SGD(model.parameters(), lr=0, momentum=0)
    score = []
        
    for i, x in enumerate(tqdm(dataloader, desc='Calculate Score GLOW', unit='step')):
        
        try: # with_label == True (ex : cifar10, svhn and etc.)
            x, _ = x
        except: # with_label == False (ex : celeba)
            pass
        
        optimizer.zero_grad()
        x = x.to(device)
        z, nll, y_logits = model(x, None)
        nll.backward()
        gradient_val = 0
        
        grads = []
        for w in dicts:
            for param in w.parameters():
                grads.append(param.grad.view(-1) ** 2)
        grads = torch.cat(grads)
        gradient_val = torch.norm(grads / Grads)
        score.append(gradient_val.detach().cpu())
        if i > max_iter:
            break
            
    return score

def AUTO_GLOW_CIFAR(model, dicts, device='cuda:0'):
    
    """ 편하게 자동화한 것입니다. """
    """ max_iter 설정은 여기 함수 내부에서 해 주세요~ """
    
    opt = config.GLOW_cifar10
    
    Grads, normalize_factor = Calculate_fisher_GLOW(model,
                                                    TRAIN_loader(option='cifar10',
                                                                 augment=True,
                                                                 is_glow=True),
                                                    dicts,
                                                    max_iter=1000,
                                                   )
    
    cifar_Gradients = Calculate_score_GLOW(model,
                                           TEST_loader(train_dist='cifar10',
                                                       target_dist='cifar10',
                                                       is_glow=True),
                                           dicts,
                                           Grads,
                                           normalize_factor,
                                           max_iter=400,
                                           with_label=opt.with_label
                                          )

    svhn_Gradients = Calculate_score_GLOW(model,
                                          TEST_loader(train_dist='cifar10',
                                                      target_dist='svhn',
                                                      is_glow=True),
                                          dicts,
                                          Grads,
                                          normalize_factor,
                                          max_iter=400,
                                          with_label=opt.with_label
                                         )

    celeba_Gradients = Calculate_score_GLOW(model,
                                            TEST_loader(train_dist='cifar10',
                                                        target_dist='celeba',
                                                        is_glow=True),
                                            dicts,
                                            Grads,
                                            normalize_factor,
                                            max_iter=400,
                                            with_label=opt.with_label
                                           )

    lsun_Gradients = Calculate_score_GLOW(model,
                                          TEST_loader(train_dist='cifar10',
                                                      target_dist='lsun',
                                                      is_glow=True),
                                          dicts,
                                          Grads,
                                          normalize_factor,
                                          max_iter=400,
                                          with_label=opt.with_label
                                         )

    noise_Gradients = Calculate_score_GLOW(model,
                                           TEST_loader(train_dist='cifar10',
                                                       target_dist='noise',
                                                       is_glow=True),
                                           dicts,
                                           Grads,
                                           normalize_factor,
                                           max_iter=400,
                                           with_label=opt.with_label
                                          )
    
    return Grads, normalize_factor, cifar_Gradients, svhn_Gradients, celeba_Gradients, lsun_Gradients, noise_Gradients


def AUTO_GLOW_FMNIST():
    
    """ 편하게 자동화한 것입니다. """
    """ max_iter 설정은 여기 함수 내부에서 해 주세요~ """
    
    return