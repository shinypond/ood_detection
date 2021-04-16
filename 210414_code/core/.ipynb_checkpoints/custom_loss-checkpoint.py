import torch
import torch.nn as nn
from torch.autograd import Variable

_loss_fn = nn.CrossEntropyLoss(reduction='none')

def KL_div(mu, logvar, reduction='avg'):
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # scalar
    else:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # vector

def VAE_loss(x, output, beta=1):
    y, mu, logvar = output
    b = x.size(0)
    target = Variable(x.data.view(-1) * 255).long()
    y = y.contiguous()
    y = y.view(-1, 256)
    recl = _loss_fn(y, target)
    recl = torch.sum(recl) / b
    kld = KL_div(mu, logvar)
    return recl + beta * kld.mean()