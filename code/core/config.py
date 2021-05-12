# config for In-Distribution (Not for OOD)
# Out-Distribution Dataset must be reformed as the below config...

class VAE_cifar10:
    train_dist = 'cifar10'
    ood_list = ['cifar10',
                'svhn',
                'celeba',
                'lsun',
                'cifar100',
                'mnist',
                'fmnist',
                'kmnist',
                'omniglot',
                'notmnist',
                #'trafficsign',
                'noise',
                'constant',
               ]
    dataroot = '../data'
    modelroot = '../saved_models'
    workers = 2
    imageSize = 32
    nc = 3   # Num of c (channels)
    nz = 200 # Num of z (latent)
    ngf = 64 # Num of Generator Filter size (scaling factor)
    ngpu = 1
    beta = 1 # Beta-VAE !
    batch_size = 1
    train_batchsize = 1
    num_samples = 20
    
class VAE_fmnist:
    train_dist = 'fmnist'
    ood_list = ['fmnist',
                'svhn',
                'celeba',
                'lsun',
                'cifar10',
                'cifar100',
                'mnist',
                'kmnist',
                'omniglot',
                'notmnist',
                'noise',
                'constant']
    dataroot = '../data'
    modelroot = '../saved_models'
    workers = 2
    imageSize = 32
    nc = 1   # Num of c (channels)
    nz = 200 # Num of z (latent)
    ngf = 32 # Num of Generator Filter size (scaling factor)
    ngpu = 1
    beta = 1 # Beta-VAE !
    batch_size = 1
    train_batchsize = 1
    num_samples = 20
    
class GLOW_cifar10:
    train_dist = 'cifar10'
    ood_list = ['cifar10',
                'svhn',
                'celeba',
                'lsun',
                #'cifar100',
                #'mnist',
                #'fmnist',
                #'kmnist',
                #'omniglot',
                #'notmnist',
                #'trafficsign',
                #'noise',
                #'constant',
               ]
    dataroot = '../data'
    modelroot = '../saved_models'
    resume_train = False
    download = False
    augment = True
    train_batchsize = 1
    eval_batchsize = 1
    epochs = 250
    seed = 0
    hidden_channels = 512
    K = 32
    L = 3
    actnorm_scale = 1.0
    flow_permutation = "invconv" # invconv, shuffle, reverse
    flow_coupling = "affine" # affine, additive
    LU_decomposed = True
    learn_top = True
    y_condition = False
    y_weight = 0.01
    max_grad_clip = 0
    max_grad_norm = 0
    lr = 5e-4
    workers = 2
    cuda = True
    n_init_batches = 8
    output_dir = f'{modelroot}/GLOW_cifar10/'
    saved_optimizer = ""
    warmup = 5
    imageSize = 32
    nc = 3
    
    
class GLOW_fmnist:
    train_dist = 'fmnist'
    ood_list = ['fmnist',
                #'svhn',
                #'celeba',
                #'lsun',
                #'cifar10',
                #'cifar100',
                'mnist',
                'kmnist',
                #'omniglot',
                'notmnist',
                'noise',
                'constant']
    dataroot = '../data'
    modelroot = '../saved_models'
    resume_train = False
    download = False
    augment = True
    train_batchsize = 1
    eval_batchsize = 1
    epochs = 250
    seed = 0
    hidden_channels = 200
    K = 8
    L = 3
    actnorm_scale = 1.0
    flow_permutation = "invconv" # invconv, shuffle, reverse
    flow_coupling = "affine" # affine, additive
    LU_decomposed = True
    learn_top = True
    y_condition = False
    y_weight = 0.01
    max_grad_clip = 0
    max_grad_norm = 0
    lr = 5e-4
    workers = 2
    cuda = True
    n_init_batches = 8
    output_dir = f'{modelroot}/GLOW_fmnist'
    saved_optimizer = ""
    warmup = 5
    imageSize = 32
    nc = 1
    
    
class CNN_fmnist:
    dataroot = '../data'
    modelroot = '../saved_models'
    workers = 1
    imageSize = 32
    nc = 1 # input image channels
    ngpu = 1
    #modelpath = '../saved_models/CNN_fmnist/cnn.pth'
    batch_size = 1
    
    
    