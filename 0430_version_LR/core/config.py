class VAE_cifar10:
    dataroot = '../data'
    workers = 2
    imageSize = 32
    nc = 3   # Num of c (channels)
    nz = 200 # Num of z (latent)
    ngf = 64 # Num of Generator Filter size (scaling factor)
    ngpu = 1
    state_E = '../saved_models/VAE_cifar10/netE_pixel_nz_200_ngf_64_beta_1.0_epoch_200.pth'
    state_G = '../saved_models/VAE_cifar10/netG_pixel_nz_200_ngf_64_beta_1.0_epoch_200.pth'
    batch_size = 1
    train_batchsize = 1
    num_samples = 20
    beta = 1
    maxiter = 1000
    with_label = True # True if len(dataset[0]) == 2
    
class VAE_fmnist:
    dataroot = '../data'
    workers = 2
    imageSize = 32
    nc = 1   # Num of c (channels)
    nz = 100 # Num of z (latent)
    ngf = 32 # Num of Generator Filter size (scaling factor)
    ngpu = 1
    state_E = '../saved_models/VAE_fmnist/netE_pixel.pth'
    state_G = '../saved_models/VAE_fmnist/netG_pixel.pth'
    batch_size = 1
    train_batchsize = 1
    num_samples = 20
    beta = 1
    with_label = True # True if len(dataset[0]) == 2
    maxiter = 10000