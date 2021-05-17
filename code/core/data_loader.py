import os
import cv2
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

import config

def TRAIN_loader(option='cifar10', shuffle=True, augment=False, is_glow=False, normalize=False, batch_size=1):
    
    """ Return train_loader for given dataset """
    """ Option : 'cifar10' or 'fmnist' """
    
    preprocess = []
    
    if is_glow:
        preprocess = [preprocess_for_glow]
    if normalize:
        preprocess = add_normalize(preprocess)
    
    if option == 'cifar10':
        if is_glow:
            opt = config.GLOW_cifar10
        else:
            opt = config.VAE_cifar10
        
        if augment:
            augment = [
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            augment = []
        
        dataset = dset.CIFAR10(
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transforms.Compose(
                augment + [
                    transforms.Resize((opt.imageSize)),
                    transforms.ToTensor(),
                ] + preprocess
            ),
        )
        train_loader_cifar = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(opt.workers),
        )
        return train_loader_cifar
        
    elif option == 'fmnist':
        if is_glow:
            opt = config.GLOW_fmnist
        else:
            opt = config.VAE_fmnist
        
        dataset = dset.FashionMNIST(
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((opt.imageSize)),
                transforms.ToTensor(),
            ] + preprocess),
        )
        train_loader_fmnist = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(opt.workers),
        )
        return train_loader_fmnist
    
    else:
        raise NotImplementationError('TRAIN distribution must be CIFAR-10 or FMNIST !')
    
    
def TEST_loader(train_dist='cifar10', target_dist='cifar10', batch_size=1, shuffle=False, is_glow=False, normalize=False):
    
    """ Return test_loader for given 'train_dist' and 'target_dist' """
    
    """ train_dist = 'cifar10' or 'fmnist' """
    
    """ target_dist (In-Distribution or Out-of-Distribution)
    
            if train_dist is 'cifar10' (train), target_dist should be one of
                    - cifar10 (test)
                    - svhn (test)     
                    - celeba (test)   
                    - lsun (test)     
                    - cifar100 (test) 
                    - mnist (test)    
                    - fmnist (test)   
                    - kmnist (test)   
                    - omniglot (eval) 
                    - notmnist (small)
                    - trafficsign
                    - noise
                    - constant
            
            if train_dist is 'fmnist' (train), target_dist should be one of
                    - fmnist (test)
                    - svhn (test)     
                    - celeba (test)   
                    - lsun (test)     
                    - cifar10 (test)  
                    - cifar100 (test) 
                    - mnist (test)    
                    - kmnist (test)   
                    - omniglot (eval) 
                    - notmnist (small)
                    - noise
                    - constant
    
    """
    
    preprocess = []
    
    if is_glow:
        preprocess = [preprocess_for_glow]
    
    if train_dist == 'cifar10':
        
        opt = config.GLOW_cifar10 if is_glow else config.VAE_cifar10
        
        if target_dist == 'cifar10':
            return test_loader_cifar10(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == 'svhn':
            return test_loader_svhn(opt, preprocess, batch_size, shuffle, normalize)

        elif target_dist == 'celeba':
            return test_loader_celeba(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'lsun':
            return test_loader_lsun(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'cifar100':
            return test_loader_cifar100(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'mnist':
            return test_loader_mnist(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'fmnist':
            return test_loader_fmnist(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'kmnist':
            return test_loader_kmnist(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'omniglot':
            return test_loader_omniglot(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'notmnist':
            return test_loader_notmnist(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'trafficsign':
            return test_loader_trafficsign(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'noise':
            return test_loader_noise(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'constant':
            return test_loader_constant(opt, preprocess, batch_size, shuffle, normalize)
        
        else:
            raise NotImplementedError("Oops! Such match of ID & OOD doesn't exist!")

    elif train_dist == 'fmnist':
        
        opt = config.GLOW_fmnist if is_glow else config.VAE_fmnist
        
        if target_dist == 'fmnist':
            return test_loader_fmnist(opt, preprocess, batch_size, shuffle, normalize)
            
        elif target_dist == 'svhn':
            return test_loader_svhn(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'celeba':
            return test_loader_celeba(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'lsun':
            return test_loader_lsun(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'cifar10':
            return test_loader_cifar10(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'cifar100':
            return test_loader_cifar100(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'mnist':
            return test_loader_mnist(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'kmnist':
            return test_loader_kmnist(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'omniglot':
            return test_loader_omniglot(opt, preprocess, batch_size, shuffle, normalize)
            
        elif target_dist == 'notmnist':
            return test_loader_notmnist(opt, preprocess, batch_size, shuffle, normalize)
            
        elif target_dist == 'noise':
            return test_loader_noise(opt, preprocess, batch_size, shuffle, normalize)
        
        elif target_dist == 'constant':
            return test_loader_constant(opt, preprocess, batch_size, shuffle, normalize)
        
        else:
            raise NotImplementedError("Oops! Such match of ID & OOD doesn't exist!")
        
    else:
        raise NotImplementedError("Oops! Such match of ID & OOD doesn't exist!")

def preprocess_for_glow(x):
    x = x * 255  # undo ToTensor scaling to [0,1]
    n_bits = 8
    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x

def rgb_to_gray(x):
    return transforms.Grayscale(1)(x)

def gray_to_rgb(x):
    return x.repeat(3, 1, 1)

def add_normalize(preprocess, nc):
    if nc == 1:
        return preprocess + [transforms.Normalize((0.48,), (0.2,))]
    elif nc == 3:
        return preprocess + [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

def test_loader_cifar10(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 1:
        preprocess += [rgb_to_gray]
    dataset_cifar10 = dset.CIFAR10(
        root=opt.dataroot,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((opt.imageSize)),
            transforms.ToTensor(),
        ] + preprocess),
    )
    test_loader_cifar10 = data.DataLoader(
        dataset_cifar10,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_cifar10

def test_loader_svhn(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 1:
        preprocess += [rgb_to_gray]
    dataset_svhn = dset.SVHN(
        root=opt.dataroot,
        split='test',
        download=True,
        transform=transforms.Compose([
            transforms.Resize((opt.imageSize,opt.imageSize)),
            transforms.ToTensor(),
        ] + preprocess),
    )
    test_loader_svhn = data.DataLoader(
        dataset_svhn,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_svhn
    
def test_loader_celeba(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 1:
        preprocess += [rgb_to_gray]
    class CelebA(data.Dataset):
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            return img

    transform=transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
    ] + preprocess)

    celeba = CelebA('../data/celeba/archive', transform=transform)
    test_loader_celeba = data.DataLoader(
        celeba,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_celeba

def test_loader_lsun(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 1:
        preprocess += [rgb_to_gray]
    class LSUN(data.Dataset):
        def __init__(self, db_path, categories=['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'], transform=None):
            super(LSUN, self).__init__()
            self.total_path = []
            for i in range(len(categories)):
                self.db_path = db_path + '/' + categories[i] + '_train'
                elements = os.listdir(self.db_path)
                self.total_path += [self.db_path + '/' + element for element in elements if element[-4:] == '.jpg']
            self.transform = transform

        def __len__(self):
            return len(self.total_path)

        def __getitem__(self, index):
            current_path = self.total_path[index]
            img = cv2.imread(current_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            return img

    transform = transforms.Compose([
        #transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.Resize(opt.imageSize), # Then the size will be H x 32 or 32 x W (32 is smaller)
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
    ] + preprocess)

    lsun = LSUN('../data/LSUN_train_10000', transform=transform)
    test_loader_lsun = data.DataLoader(
        lsun,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_lsun

def test_loader_cifar100(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 1:
        preprocess += [rgb_to_gray]
    dataset_cifar100 = dset.CIFAR100(
        root=opt.dataroot,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((opt.imageSize)),
            transforms.ToTensor(),
        ] + preprocess),
    )
    test_loader_cifar100 = data.DataLoader(
        dataset_cifar100,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_cifar100

def test_loader_mnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 3:
        preprocess += [gray_to_rgb]
    dataset_mnist = dset.MNIST(
        root=opt.dataroot,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ] + preprocess),
    )
    test_loader_mnist = data.DataLoader(
        dataset_mnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_mnist
    
def test_loader_fmnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 3:
        preprocess += [gray_to_rgb]
    dataset_fmnist = dset.FashionMNIST(
        root=opt.dataroot,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((opt.imageSize)),
            transforms.ToTensor(),
        ] + preprocess),
    )
    test_loader_fmnist = data.DataLoader(
        dataset_fmnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_fmnist
    
def test_loader_kmnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 3:
        preprocess += [gray_to_rgb]
    dataset_kmnist = dset.KMNIST(
        root=opt.dataroot,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ] + preprocess),
    )
    test_loader_kmnist = data.DataLoader(
        dataset_kmnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_kmnist
    
def test_loader_omniglot(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 3:
        preprocess += [gray_to_rgb]
    dataset_omniglot = dset.Omniglot(
        root=opt.dataroot, 
        background=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ] + preprocess),
    )
    test_loader_omniglot = data.DataLoader(
        dataset_omniglot,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
    )
    return test_loader_omniglot
    
def test_loader_notmnist(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 1:
        preprocess += [rgb_to_gray]
    class notMNIST(data.Dataset):
        def __init__(self, db_path, transform=None):
            super(notMNIST, self).__init__()
            self.db_path = db_path
            self.total_path = []
            alphabets = os.listdir(self.db_path)
            for alphabet in alphabets:
                path = self.db_path + '/' + alphabet
                elements = os.listdir(path)
                self.total_path += [path + '/' + element for element in elements]
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

    notmnist = notMNIST('../data/notMNIST_small/', transform=transform)
    test_loader_notmnist = data.DataLoader(
        notmnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_notmnist
    
def test_loader_trafficsign(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    if opt.nc == 1:
        preprocess += [rgb_to_gray]
    class trafficsign(data.Dataset):
        def __init__(self, db_path, transform=None):
            super(trafficsign, self).__init__()
            self.db_path = db_path
            elements = os.listdir(self.db_path)
            self.total_path = [self.db_path + '/' + element for element in elements]
            self.transform = transform

        def __len__(self):
            return len(self.total_path)

        def __getitem__(self, index):
            current_path = self.total_path[index]
            img = cv2.imread(current_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            return img

    transform = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
    ] + preprocess)

    ts = trafficsign('../data/GTSRB_Final_Test_Images/Final_Test/Images', transform=transform)
    test_loader_trafficsign = data.DataLoader(
        ts,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_trafficsign
    
def test_loader_noise(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    class Noise(data.Dataset):
        def __init__(self, number=10000, transform=None):
            super(Noise, self).__init__()
            self.transform = transform
            self.number = number
            self.total_data = np.random.randint(0, 256, (self.number, opt.nc, 32, 32))

        def __len__(self):
            return self.number

        def __getitem__(self, index):
            array = torch.tensor(self.total_data[index] / 255).float()
            return self.transform(array)

    transform = transforms.Compose(preprocess)

    noise = Noise(transform=transform)
    test_loader_noise = data.DataLoader(
        noise,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_noise
    
def test_loader_constant(opt, preprocess, batch_size, shuffle, normalize=False):
    if normalize:
        preprocess = add_normalize(preprocess, opt.nc)
    class Constant(data.Dataset):
        def __init__(self, number=10000, transform=None):
            super(Constant, self).__init__()
            self.number = number
            self.total_data = np.random.randint(0, 256, (self.number, opt.nc, 1, 1))
            self.transform = transform

        def __len__(self):
            return self.number

        def __getitem__(self, index):
            array = torch.tensor(self.total_data[index] / 255).float()
            array = array.repeat(1, 32, 32)
            return self.transform(array)

    transform = transforms.Compose(preprocess)

    constant = Constant(transform=transform)
    test_loader_constant = data.DataLoader(
        constant,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return test_loader_constant