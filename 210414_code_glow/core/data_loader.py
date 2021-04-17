import os
import cv2
from PIL import Image
import numpy as np

import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

import config

def TRAIN_loader(option='cifar10', shuffle=True, augment=False, is_glow=False):
    
    """ Return train_loader for given dataset """
    
    """ Option : 'cifar10' or 'fmnist' """
    
    preprocess = []
    if is_glow:
        preprocess = [preprocess_for_glow]
    
    if option == 'cifar10':
        
        if is_glow:
            opt = config.GLOW_cifar10
        else:
            opt = config.VAE_cifar10
        
        if augment:
            augment = [
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            augment = []
        
        dataset = dset.CIFAR10(root=opt.dataroot,
                               train=True,
                               download=True,
                               transform=transforms.Compose(
                                   augment +
                                   [transforms.Resize((opt.imageSize)),
                                    transforms.ToTensor(),]
                                   + preprocess))
        
        train_loader_cifar = torch.utils.data.DataLoader(dataset,
                                                         batch_size=opt.train_batchsize,
                                                         shuffle=shuffle,
                                                         num_workers=int(opt.workers))
        return train_loader_cifar
        
    elif option == 'fmnist':
        
        if is_glow:
            opt = config.GLOW_fmnist # 에러날 거
        else:
            opt = config.VAE_fmnist
        
        dataset = dset.FashionMNIST(root=opt.dataroot,
                                    train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize((opt.imageSize)),
                                        transforms.ToTensor(),
                                    ] + preprocess))
        
        train_loader_fmnist = torch.utils.data.DataLoader(dataset,
                                                          batch_size=opt.train_batchsize,
                                                          shuffle=shuffle,
                                                          num_workers=int(opt.workers))
        return train_loader_fmnist
    

def TEST_loader(train_dist='cifar10', target_dist='cifar10', batch_size=1, shuffle=False, is_glow=False):
    
    """ Return test_loader for given 'train_dist' and 'target_dist' """
    
    """ train_dist = 'cifar10' or 'fmnist' """
    
    """ target_dist (In-Distribution or Out-of-Distribution)
    
            if train_dist is 'cifar10', target_dist should be one of
                (1) cifar10 (test) [ID]
                (2) svhn (test)    [OOD]
                (3) celeba (test)  [OOD]
                (4) lsun (test)    [OOD]
                (5) noise          [OOD]
            
            if train_dist is 'fmnist', target_dist should be one of
                (1) fmnist (test)  [ID]
                (2) mnist (test)   [OOD]
                (3) noise          [OOD]
    
    """
    
    preprocess = []
    if is_glow:
        preprocess = [preprocess_for_glow]
    
    if train_dist == 'cifar10':
        
        if is_glow:
            opt = config.GLOW_cifar10
        else:
            opt = config.VAE_cifar10
        
        if target_dist == 'cifar10':
            dataset_cifar10 = dset.CIFAR10(root=opt.dataroot,
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((opt.imageSize)),
                                               transforms.ToTensor(),
                                           ] + preprocess))
            
            test_loader_cifar = torch.utils.data.DataLoader(dataset_cifar10,
                                                            batch_size=batch_size,
                                                            shuffle=shuffle,
                                                            num_workers=int(opt.workers))
            return test_loader_cifar

        elif target_dist == 'svhn':
            dataset_svhn = dset.SVHN(root=opt.dataroot,
                                     split='test',
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize((opt.imageSize,opt.imageSize)),
                                         transforms.ToTensor(),
                                     ] + preprocess))
            
            test_loader_svhn = torch.utils.data.DataLoader(dataset_svhn,
                                                           batch_size=batch_size,
                                                           shuffle=shuffle,
                                                           num_workers=int(opt.workers))
            return test_loader_svhn

        elif target_dist == 'celeba':
            class CelebA(torch.utils.data.Dataset):

                def __init__(self, db_path, transform=None):
                    super(CelebA, self).__init__()
                    self.db_path = db_path
                    elements = os.listdir(self.db_path)
                    self.total_path = [self.db_path + '/' + element for element in elements]
                    self.transform = transform

                def __len__(self):
                    return len(self.total_path)

                def __getitem__(self, index):
                    current_path = self.total_path[index]
                    img = cv2.imread(current_path)
                    img = Image.fromarray(img)
                    img = self.transform(img)
                    return img

            transform=transforms.Compose([
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ] + preprocess)

            celeba = CelebA('../data/celeba/archive', transform=transform)
            test_loader_celeba = torch.utils.data.DataLoader(celeba,
                                                             batch_size=batch_size,
                                                             shuffle=shuffle,
                                                             num_workers=0)
            return test_loader_celeba
        
        elif target_dist == 'lsun':
            class LSUN(torch.utils.data.Dataset):
                
                def __init__(self, db_path, categories=['bedroom', 'kitchen'], transform=None):
                    super(LSUN, self).__init__()
                    self.total_path = []
                    for i in range(len(categories)):
                        self.db_path = db_path + '/' + categories[i] + '_val'
                        elements = os.listdir(self.db_path)
                        self.total_path += [self.db_path + '/' + element for element in elements]
                    self.transform = transform
                    
                def __len__(self):
                    return len(self.total_path)
                
                def __getitem__(self, index):
                    current_path = self.total_path[index]
                    img = cv2.imread(current_path)
                    img = Image.fromarray(img)
                    img = self.transform(img)
                    return img
                
            transform = transforms.Compose([
                transforms.Resize((opt.imageSize, opt.imageSize)),
                transforms.ToTensor(),
            ] + preprocess)
                    
            lsun = LSUN('../data/LSUN_test', transform=transform)
            test_loader_lsun = torch.utils.data.DataLoader(lsun,
                                                           batch_size=batch_size,
                                                           shuffle=shuffle,
                                                           num_workers=0)
            return test_loader_lsun
        
        elif target_dist == 'noise':
            class Noise(torch.utils.data.Dataset):
                def __init__(self, number=10000, transform=None):
                    super(Noise, self).__init__()
                    self.transform = transform
                    self.number = number
                    self.total_data = np.random.randint(0, 256, (self.number, 3, 32, 32))

                def __len__(self):
                    return self.number

                def __getitem__(self, index):
                    if self.transform is not None:
                        return self.transform(self.total_data[index])
                    else:
                        return np.array(self.total_data[index] / 255, dtype='float32')

            noise = Noise()
            test_loader_noise = torch.utils.data.DataLoader(noise,
                                                            batch_size=batch_size,
                                                            shuffle=shuffle,
                                                            num_workers=0)
            return test_loader_noise
        
        else:
            print("Oops! Such match of ID & OOD doesn't exist!")
            raise NotImplementedError

    elif train_dist == 'fmnist':
        
        if is_glow:
            opt = config.GLOW_fmnist # 에러날 거
        else:
            opt = config.VAE_fmnist
        
        if target_dist == 'fmnist':
            dataset_fmnist = dset.FashionMNIST(root=opt.dataroot,
                                               train=False,
                                               download=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.imageSize)),
                                                   transforms.ToTensor(),
                                               ] + preprocess))
            
            test_loader_fmnist = torch.utils.data.DataLoader(dataset_fmnist,
                                                             batch_size=batch_size,
                                                             shuffle=shuffle,
                                                             num_workers=int(opt.workers))
            return test_loader_fmnist
            
        elif target_dist == 'mnist':
            dataset_mnist = dset.MNIST(root=opt.dataroot,
                                       train=False,
                                       download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize((opt.imageSize, opt.imageSize)),
                                           transforms.ToTensor(),
                                       ] + preprocess))
            
            test_loader_mnist = torch.utils.data.DataLoader(dataset_mnist,
                                                            batch_size=batch_size,
                                                            shuffle=shuffle,
                                                            num_workers=int(opt.workers))
            return test_loader_mnist
        
        elif target_dist == 'noise':
            class Noise(torch.utils.data.Dataset):
                def __init__(self, number=10000, transform=None):
                    super(Noise, self).__init__()
                    self.transform = transform
                    self.number = number
                    self.total_data = np.random.randint(0, 256, (self.number, 1, 32, 32))

                def __len__(self):
                    return self.number

                def __getitem__(self, index):
                    if self.transform is not None:
                        return self.transform(self.total_data[index])
                    else:
                        return np.array(self.total_data[index] / 255, dtype='float32')

            noise = Noise()
            test_loader_noise = torch.utils.data.DataLoader(noise,
                                                            batch_size=batch_size,
                                                            shuffle=shuffle,
                                                            num_workers=0)
            return test_loader_noise
        
        else:
            print("Oops! Such match of ID & OOD doesn't exist!")
            raise NotImplementedError
        
    else:
        print("Oops! Such match of ID & OOD doesn't exist!")
        raise NotImplementedError
        
def preprocess_for_glow(x):
    
    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bits = 8
    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x
    