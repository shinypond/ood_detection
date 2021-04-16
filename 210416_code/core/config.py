# config for In-Distribution (Not for OOD)
# Out-Distribution Dataset must be reformed as the below config...

class VAE_cifar10:
    train_dist = 'cifar10'
    ood_list = ['cifar10', 'svhn', 'celeba', 'lsun', 'noise']
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
    ood_list = ['fmnist', 'mnist', 'noise']
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
    K = 32
    L = 3
    LU_decomposed = True
    actnorm_scale = 1
    augment = True
    train_batchsize = 1
    cuda = True
    dataroot = "../data/"
    dataset = "cifar10"
    download = False
    epochs = 1500
    eval_batch_size = 512
    flow_coupling = "affine"
    flow_permutation = "invconv"
    fresh = True
    hidden_channels = 512
    learn_top = True
    lr = 0.0005
    max_grad_clip = 0
    max_grad_norm = 0
    n_init_batches = 8
    workers = 2
    output_dir = "output/"
    saved_model = "../saved_models/GLOW_cifar10/glow_affine_coupling.pt"
    saved_optimizer = ""
    seed = 0
    warmup_steps = 4000
    y_condition = False
    y_weight = 0.01
    with_label = True # True if len(dataset[0]) == 2
    imageSize = 32