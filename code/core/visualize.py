import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn import metrics

def plot_hist(*args, bins=[100, 100], labels=['cifar10', 'svhn'], xlim=[0, 10]):
    
    if len(args) != len(bins) or len(args) != len(labels) or len(bins) != len(labels):
        print('Oops! GRADS, BINS and LABELS must have the same length !')
        raise NotImplementedError
    
    fig = plt.figure(figsize=(12, 9))
    
    in_dist_Grads = torch.tensor(args[0]).detach().cpu().numpy()
    plt.hist(in_dist_Grads, bins=bins[0], density=True, alpha=0.9, color='black', label=labels[0])

    for i in range(1, len(args)):
        out_dist_Grads = torch.tensor(args[i]).detach().cpu().numpy()
        plt.hist(out_dist_Grads, bins=bins[i], density=True, alpha=0.5, label=labels[i])
    
    plt.xlim(xlim[0], xlim[1])
    plt.title(f'In-Dist : {labels[0]}  /  Out-Dist : {[labels[i] for i in range(1, len(args))]}')
    plt.grid(True)
    plt.legend()
    plt.show()

def AUROC(*args, labels=['cifar10', 'svhn'], verbose=True):    
    
    in_dist_Grads = args[0]
    out_dist_Grads = args[1]
    combined = np.concatenate((in_dist_Grads, out_dist_Grads))
    label_1 = np.ones(len(in_dist_Grads))
    label_2 = np.zeros(len(out_dist_Grads))
    label = np.concatenate((label_1, label_2))
    fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
    #plot_roc_curve(fpr, tpr)
    rocauc = metrics.auc(fpr, tpr)
    if verbose:
        title = f'In-dist : {labels[0]}  /  Out-dist : {labels[1]} \n AUC is: {rocauc:.6f}'
        fig = plt.figure(figsize=(12, 9))
        plt.plot(fpr, tpr)
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    return rocauc

def plot_scores_all_layers(train_dist, params, SCOREs, opt, save=True):
    auroc = {}
    for pname in params.keys():
        _auroc = {}
        for ood in opt.ood_list:
            args = [
                SCOREs['VAE'][train_dist][train_dist][pname],
                SCOREs['VAE'][train_dist][ood][pname],
            ]
            labels = [train_dist, ood]
            _auroc[ood] = AUROC(*args, labels=labels, verbose=False)
        auroc[pname] = _auroc
   
    fig = plt.figure(figsize=(60, 55))
    plt.subplots_adjust(wspace=0.1)
    plt.subplots_adjust(hspace=0.2)
    for i, ood in enumerate(opt.ood_list):
        df = pd.DataFrame(SCOREs['VAE'][train_dist][ood])
        df.loc['mean', :] = df.mean()
        df.loc['std', :] = df.std()
        df.loc['max', :] = df.max()
        df.loc['min', :] = df.min()
        ax = fig.add_subplot(len(opt.ood_list), 3, 3*i+1)
        ax.set_title(f'{ood}')
        ax.errorbar(df.columns, df.loc['mean', :], df.loc['std', :], linestyle='None', marker='^', color='g')
        ax.scatter(df.columns, df.loc['max', :], color='r')
        ax.scatter(df.columns, df.loc['min', :], color='r')
        ax.grid()
    for i, ood in enumerate(opt.ood_list):
        df = pd.DataFrame(SCOREs['VAE'][train_dist][ood])
        df.loc['mean', :] = df.mean()
        ax = fig.add_subplot(len(opt.ood_list), 3, 3*i+2)
        ax.set_title(f'{ood}')
        ax.plot(df.columns, df.loc['mean', :])
        ax.grid()
    df = pd.DataFrame(auroc)
    for i, ood in enumerate(opt.ood_list):
        ax = fig.add_subplot(len(opt.ood_list), 3, 3*i+3)
        ax.set_title(f'{ood}')
        ax.plot(df.columns, df.loc[ood, :], color='r')
        ax.grid()

    if save:
        fig.savefig('./score_mean_std_auroc.png')
    
    plt.show()