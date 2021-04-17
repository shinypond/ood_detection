# config for In-Distribution (Not for OOD)
# Out-Distribution Dataset must be reformed as the below config...

class VAE_cifar10:
    dataroot = '../data'
    workers = 2
    imageSize = 32
    nc = 3   # Num of c (channels)
    nz = 200 # Num of z (latent)
    ngf = 64 # Num of Generator Filter size (scaling factor)
    ngpu = 1
    state_E = f'../saved_models/VAE_cifar10/netE_pixel_nz_{nz}_ngf_{ngf}_epoch_200.pth'
    state_G = f'../saved_models/VAE_cifar10/netG_pixel_nz_{nz}_ngf_{ngf}_epoch_200.pth'
    batch_size = 1
    train_batchsize = 1
    num_samples = 20
    beta = 1 # Beta-VAE !
    with_label = True # True if len(dataset[0]) == 2
    
    
class VAE_fmnist:
    dataroot = '../data'
    workers = 2
    imageSize = 32
    nc = 1   # Num of c (channels)
    nz = 200 # Num of z (latent)
    ngf = 32 # Num of Generator Filter size (scaling factor)
    ngpu = 1
    state_E = f'../saved_models/VAE_fmnist/netE_pixel_nz_{nz}_ngf_{ngf}_epoch_200.pth'
    state_G = f'../saved_models/VAE_fmnist/netG_pixel_nz_{nz}_ngf_{ngf}_epoch_200.pth'
    batch_size = 1
    train_batchsize = 1
    num_samples = 20
    beta = 1 # Beta-VAE !
    with_label = True # True if len(dataset[0]) == 2
    
# class GLOW_cifar10:
#     K = 32
#     L = 3
#     LU_decomposed = True
#     actnorm_scale = 1
#     augment = True
#     train_batchsize = 1
#     cuda = True
#     dataroot = "../data/"
#     dataset = "cifar10"
#     download = False
#     epochs = 1500
#     eval_batch_size = 512
#     flow_coupling = "affine"
#     flow_permutation = "invconv"
#     fresh = True
#     hidden_channels = 512
#     learn_top = True
#     lr = 0.0005
#     max_grad_clip = 0
#     max_grad_norm = 0
#     n_init_batches = 8
#     workers = 2
#     output_dir = "output/"
#     saved_model = "../saved_models/GLOW_cifar10/glow_affine_coupling.pt"
#     saved_optimizer = ""
#     seed = 0
#     warmup_steps = 4000
#     y_condition = False
#     y_weight = 0.01
#     with_label = True # True if len(dataset[0]) == 2
#     imageSize = 32
    
    
class GLOW_cifar10:
    dataset = "cifar10" # cifar10, fmnist
    dataroot = "../../../data/"
    download = False
    augment = True
    train_batchsize = 32
    eval_batch_size = 512
    epochs = 250
#     saved_model = "../../../saved_models/GLOW_cifar10/glow_affine_coupling.pt"
    saved_model = ""
    seed = 0
    hidden_channels = 400
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
    output_dir = "output_cifar10/"
    saved_optimizer = ""
    warmup = 5
    imageSize = 32
    
    
class GLOW_fmnist:
    dataset = "fmnist"
    dataroot = "../../../data/"
    download = False
    augment = True
    train_batchsize = 32
    eval_batch_size = 512
    epochs = 250
#     saved_model = "../../../saved_models/GLOW_fmnist/glow_affine_coupling.pt"
    saved_model = ""
    seed = 0
    hidden_channels = 200
    K = 16
    L = 2
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
    output_dir = "output_fmnist/"
    saved_optimizer = ""
    warmup = 5
    imageSize = 28