import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from train_GLOW.model import Glow
from data_loader import TRAIN_loader, TEST_loader
import config

# fix a random seed
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

if __name__ == '__main__':

    print('Q. Dataset?')
    print('(1) CIFAR-10 (2) FMNIST')
    ans = input('-> ')
    if ans == '1':
        opt = config.GLOW_cifar10
        image_shape = (32, 32, 3)
        num_classes = 10
    elif ans == '2':
        opt = config.GLOW_fmnist
        image_shape = (32, 32, 1)
        num_classes = 10
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Glow(
        image_shape,
        opt.hidden_channels,
        opt.K,
        opt.L,
        opt.actnorm_scale,
        opt.flow_permutation,
        opt.flow_coupling,
        opt.LU_decomposed,
        num_classes,
        opt.learn_top,
        opt.y_condition,
    )
    model_path = f'../{opt.modelroot}/GLOW_{opt.train_dist}/glow_{opt.train_dist}.pt'
    model.load_state_dict(torch.load(model_path)['model'])
    model.set_actnorm_init()
    model = model.to(device)
    model.eval()
    start = datetime.now()
    
    for ood in opt.ood_list:
    
        torch.manual_seed(2021)
        random.seed(2021)
        test_loader = TEST_loader(opt.train_dist, ood, shuffle=True, is_glow=True, normalize=False)
        NLL = []
        
        for i, x in enumerate(test_loader):
            try:
                x, _ = x
            except:
                pass
            
            x = x.to(device)
            with torch.no_grad():
                _, nll, _ = model(x, None)
            NLL = np.append(NLL, nll.item())
            print(f'[ {i+1:04d} / 5000 ] {opt.train_dist}/{ood} NLL running_mean {NLL.mean():.2f} Elapsed time {datetime.now() - start}')
            
            if i >= 4999:
                break
        np.save(f'../GLOW_NLL_npy/{opt.train_dist}_{ood}_nll.npy', NLL)
            
        