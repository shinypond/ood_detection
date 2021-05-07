import torch
import torch.nn as nn
from torch.autograd import Variable

_loss_fn_pixel = nn.CrossEntropyLoss(reduction='none')
_loss_fn_rgb = nn.BCELoss(reduction='sum')

def KL_div(mu, logvar, reduction='avg'):
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == 'sum':
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # scalar
    else:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # vector

def VAE_loss_pixel(x, output, beta=1):
    y, mu, logvar = output
    b = x.size(0)
    target = Variable(x.data.view(-1) * 255).long()
    y = y.contiguous()
    y = y.view(-1, 256)
    recl = _loss_fn_pixel(y, target)
    recl = torch.sum(recl) / b
    kld = KL_div(mu, logvar)
    return recl + beta * kld.mean()

def loglikelihood(x, z, kernel_output):
    y, mu, logvar = kernel_output
    b = x.size(0)
    target = Variable(x.data.view(-1) * 255).long()
    y = y.contiguous()
    y = y.view(-1,256)
    recl = _loss_fn_pixel(y, target)
    recl = recl.view(b,-1)
    var = logvar.exp()
    recl_loglikelihood = -recl.sum(1)
    log_ratio = \
        logvar.sum(1).squeeze() / 2 \
        + ((z - mu) ** 2 / (2 * var)).sum(1).squeeze() \
        - (z ** 2).sum(1).squeeze() / 2
    loglikelihood = recl_loglikelihood + log_ratio
    loglikelihood = torch.logsumexp(loglikelihood, 0)
    return - loglikelihood
    