import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from custom_loss import VAE_loss_pixel, loglikelihood
import config
from data_loader import TRAIN_loader, TEST_loader
from model_loader import load_pretrained_VAE
from fisher_utils_VAE import Calculate_fisher_VAE, Calculate_score_VAE

# fix a random seed
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)


opt = config.VAE_cifar10
netE, netG = load_pretrained_VAE(option=opt.train_dist, ngf=64, nz=200, beta=1, augment='hflip', epoch=100)
netE.eval()
netG.eval()

params = {
    'Econv1_w': netE.conv1.weight,
}

method = 'SMW'
max_iter = 5000

loss_type = 'ELBO_pixel'
########################################################################
print('Start to get Fisher inv')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
netE.eval()
netG.eval()
optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
Fisher_inv = {}
normalize_factor = {}
grads = {}
count = 0
train_dataloader = TRAIN_loader(
    option=opt.train_dist,
    shuffle=True,
    is_glow=False,
)

for i, (x, _) in enumerate(train_dataloader):

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
    [z, mu, logvar] = netE(x)

    if opt.num_samples == 1:
        recon = netG(mu)
    elif opt.num_samples > 1:
        recon = netG(z)
    else:
        raise ValueError

    recon = recon.contiguous()

    if loss_type == 'ELBO_pixel':
        loss = VAE_loss_pixel(x, [recon, mu, logvar])
    elif loss_type == 'exact':
        loss = loglikelihood(x, z, [recon, mu, logvar])
    else:
        raise ValueError

    loss.backward(retain_graph=True)

    if method == 'exact':
        count += 1
        for pname, param in params.items():
            grads[pname] = []
            for j in range(param.grad.shape[0]):
                grads[pname].append(param.grad[j, :, :, :].view(-1, 1))
            grads[pname] = torch.cat(grads[pname], dim=1).T.to(device)
            grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1) # 800 x 1024

            if i == 0:
                identity = torch.diag(torch.ones(grads[pname].shape[1]))
                Fisher_inv[pname] = 1e-3 * identity.unsqueeze(0).repeat(grads[pname].shape[0], 1, 1).to(device)
            Fisher_inv[pname] += torch.bmm(grads[pname].unsqueeze(2), grads[pname].unsqueeze(1))

    elif method == 'SMW':
        count += 1
        for pname, param in params.items():
            grads[pname] = []
            for j in range(param.grad.shape[0]):
                grads[pname].append(param.grad[j, :, :, :].view(-1, 1))
            grads[pname] = torch.cat(grads[pname], dim=1).T.to(device)
            grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1)

            if i == 0:
                Fisher_inv[pname] = 1000 * torch.diag(torch.ones(grads[pname].shape[1])).unsqueeze(0).to(device)
                Fisher_inv[pname] = Fisher_inv[pname].repeat(grads[pname].shape[0], 1, 1)

            u1 = grads[pname].unsqueeze(1)
            u2 = grads[pname].unsqueeze(2)
            b = torch.bmm(Fisher_inv[pname], u2)
            denom = torch.ones(grads[pname].shape[0], 1).to(device) + torch.bmm(u1, b).squeeze(2)
            denom = denom.unsqueeze(2)
            numer = torch.bmm(b, b.permute(0, 2, 1))
            Fisher_inv[pname] -= numer / denom

    elif method == 'Vanilla':
        for pname, param in params.items():
            grads[pname] = param.grad.view(-1) ** 2
            if i == 0:
                Fisher_inv[pname] = grads[pname]
            else:
                Fisher_inv[pname] = (i * Fisher_inv[pname] + grads[pname]) / (i + 1)

    if i >= 10000 - 1:
        break

if method == 'exact':
    for pname, _ in params.items():
        for j in range(Fisher_inv[pname].shape[0]):
            Fisher_inv[pname][j, :, :] = count * torch.inverse(Fisher_inv[pname][j, :, :])
            normalize_factor[pname] = 2 * np.sqrt(np.array(Fisher_inv[pname].shape).prod())

elif method == 'SMW':
    for pname, _ in params.items():
        Fisher_inv[pname] *= count
        normalize_factor[pname] = 2 * np.sqrt(np.array(Fisher_inv[pname].shape).prod())

elif method == 'Vanilla':
    for pname, _ in params.items():
        Fisher_inv[pname] = torch.sqrt(Fisher_inv[pname])
        Fisher_inv[pname] = Fisher_inv[pname] * (Fisher_inv[pname] > 1e-3)
        Fisher_inv[pname][Fisher_inv[pname]==0] = 1e-3
        normalize_factor[pname] = 2 * np.sqrt(len(Fisher_inv[pname]))


#############################################################
print('Start to get In-dist score')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
netE.eval()
netG.eval()
optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
grads = {}
score = {}
id_dataloader = TEST_loader(
    train_dist=opt.train_dist,
    target_dist=opt.train_dist,
    shuffle=True,
    is_glow=False,
)

for i, x in enumerate(id_dataloader):

    try: # with label (ex : cifar10, svhn and etc.)
        x, _ = x
    except: # without label (ex : celeba)
        pass

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
    gradient_val = 0
    [z, mu, logvar] = netE(x)
    recon = netG(z)
    recon = recon.contiguous()
    loss = VAE_loss_pixel(x, [recon, mu, logvar])
    loss.backward(retain_graph=True)

    if method == 'SMW' or method == 'exact':
        for pname, param in params.items():
            grads[pname] = []
            for j in range(param.grad.shape[0]):
                grads[pname].append(param.grad[j, :, :, :].view(-1, 1)) # 4096 x 1
            grads[pname] = torch.cat(grads[pname], dim=1).T.to(device) # 200 x 4096
            grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1) # (200 x 64) x (4096 / 64)
            u1 = grads[pname].unsqueeze(1)
            u2 = grads[pname].unsqueeze(2)
            s = torch.bmm(torch.bmm(u1, Fisher_inv[pname]), u2)
            s = torch.sum(s).detach().cpu().numpy()
            if i == 0:
                score[pname] = []
            score[pname].append(s)

    elif method == 'Vanilla':
        for pname, param in params.items():
            grads[pname] = param.grad.view(-1)
            s = torch.norm(grads[pname] / Fisher_inv[pname]).detach().cpu()
            if i == 0:
                score[pname] = []
            score[pname].append(s.numpy())
            
    if i >= max_iter - 1:
        break

id_score = np.array(score['Econv1_w']) / normalize_factor['Econv1_w']

#############################################################
print('Start to get OOD score')
netE.eval()
netG.eval()
optimizer1 = optim.SGD(netE.parameters(), lr=0, momentum=0) # no learning
optimizer2 = optim.SGD(netG.parameters(), lr=0, momentum=0) # no learning
grads = {}
score = {}
dataloaders = []
for ood in opt.ood_list:
    if ood != opt.train_dist:
        loader = TEST_loader(
            train_dist=opt.train_dist,
            target_dist=ood,
            shuffle=True,
            is_glow=False,
        )
        dataloaders.append(loader)
i = 0
loader_idx_list = np.random.randint(0, len(dataloaders), size=max_iter)

while i < max_iter:

    x = next(iter(dataloaders[loader_idx_list[i]]))
    ood = opt.ood_list[loader_idx_list[i]+1]
    print(f'Selected OOD : {ood}')
    try: # with label (ex : cifar10, svhn and etc.)
        x, _ = x
    except: # without label (ex : celeba)
        pass

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    x = x.repeat(opt.num_samples, 1, 1, 1).to(device)
    gradient_val = 0
    [z, mu, logvar] = netE(x)
    recon = netG(z)
    recon = recon.contiguous()
    loss = VAE_loss_pixel(x, [recon, mu, logvar])
    loss.backward(retain_graph=True)

    if method == 'SMW' or method == 'exact':
        for pname, param in params.items():
            grads[pname] = []
            for j in range(param.grad.shape[0]):
                grads[pname].append(param.grad[j, :, :, :].view(-1, 1)) # 4096 x 1
            grads[pname] = torch.cat(grads[pname], dim=1).T.to(device) # 200 x 4096
            grads[pname] = grads[pname].reshape(grads[pname].shape[0] * 4, -1) # (200 x 64) x (4096 / 64)
            u1 = grads[pname].unsqueeze(1)
            u2 = grads[pname].unsqueeze(2)
            s = torch.bmm(torch.bmm(u1, Fisher_inv[pname]), u2)
            s = torch.sum(s).detach().cpu().numpy()
            if i == 0:
                score[pname] = []
            score[pname].append(s)

    elif method == 'Vanilla':
        for pname, param in params.items():
            grads[pname] = param.grad.view(-1)
            s = torch.norm(grads[pname] / Fisher_inv[pname]).detach().cpu()
            if i == 0:
                score[pname] = []
            score[pname].append(s.numpy())

    i += 1
        
ood_score = np.array(score['Econv1_w']) / normalize_factor['Econv1_w']

#############################################################

indist_ = np.array(id_score)
ood_ = np.array(ood_score)
combined = np.concatenate((indist_, ood_))
label_1 = np.ones(len(indist_))
label_2 = np.zeros(len(ood_))
label = np.concatenate((label_1, label_2))
fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
aucroc = metrics.auc(fpr, tpr)
print(aucroc)














