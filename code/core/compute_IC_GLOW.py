import os, sys, argparse, random, copy
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
import cv2
import numpy as np
from datetime import datetime
from datetime import timedelta
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

sys.path.append(os.getcwd() + '/core')

from data_loader import TRAIN_loader, TEST_loader
from train_GLOW.model_GLOW import Glow
import config

# fix a random seed
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** 8
    return torch.clamp(x, 0, 255).byte()


if __name__=="__main__":

    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Q. Dataset?')
    print('(1) CIFAR-10 (2) FMNIST')
    ans = input('-> ')
    
    if ans == '1':
        opt = config.GLOW_cifar10
        size = 32 * 32 * 3
        image_shape = (32, 32, 3)
        num_classes = 10
        split = True
        decay = '5e-05'
        epoch = 50
        end = '.pt'
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
            split,
        )
        model_path = f'{opt.modelroot}/GLOW_{opt.train_dist}/glow_{opt.train_dist}_decay_{decay}_epoch_{epoch}{end}'
        model.load_state_dict(torch.load(model_path)['model']) # there are two keys: 'model', 'optimizer'
        model.set_actnorm_init()
        model = model.to(device)
        model = model.eval()
        
    elif ans == '2':
        opt = config.GLOW_fmnist
        size = 32 * 32
        image_shape = (32, 32, 1)
        num_classes = 10
        split = False
        decay = '0'
        epoch = 50
        end = '.pth'
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
            split,
        )
        model_path = f'{opt.modelroot}/GLOW_{opt.train_dist}/glow_{opt.train_dist}_decay_{decay}_epoch_{epoch}{end}'
        model.load_state_dict(torch.load(model_path))
        model.set_actnorm_init()
        model = model.to(device)
        model = model.eval()
        
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    assert opt.dataroot == '../../data', 'please go config.py and modify dataroot to "../../data"'
    
    print('Q. IC type?')
    print('(1) png (2) jp2')
    ans = input('-> ')
    if ans == '1':
        opt.ic_type = 'png'
    elif ans == '2':
        opt.ic_type = 'jp2'
    else:
        raise ValueError('Insert 1 or 2. Bye.')
        
    print('Q. Test Num?')
    test_num = int(input('-> '))

    ######################################################################
    
    #for ood in opt.ood_list:
    for ood in opt.ood_list:

        torch.manual_seed(2021)
        random.seed(2021)
        dataloader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=True,
        )
        
        Complexity = []
        difference = []
        
        for i, x in enumerate(dataloader):
            try:
                x, _ = x
            except:
                pass
            
            with torch.no_grad():
                x = x.to(device)
                _, NLL_loss, _ = model(x, None)
                NLL_loss = NLL_loss.detach().cpu().numpy()[0]
                img = postprocess(x[0]).permute(1, 2, 0) # img *= 255
                img = img.detach().cpu().numpy() 
                img = img.astype(np.uint8)
                if opt.ic_type == 'jp2':
                    img_encoded = cv2.imencode('.jp2',img)
                elif opt.ic_type == 'png':
                    img_encoded = cv2.imencode('.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                else:
                    assert 0==1
                L = len(img_encoded[1]) * 8
                Complexity.append(L)
                difference.append(NLL_loss * (size) * np.log(2) - L)
                
            
                print(f'{ood} GLOW: image {i} IC loss {NLL_loss * (size) * np.log(2)  - L:.2f}')
            
            if i >= test_num - 1:
                break
                
        difference = np.asarray(difference)

        np.save(f'../npy/IC({opt.ic_type})_GLOW/{opt.train_dist}_{ood}.npy', difference)
        print(f'saved {opt.train_dist}_{ood} IC({opt.ic_type})_GLOW npy !')
