import matplotlib.pyplot as plt
import torch
import numpy as np
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

def AUROC(*args, labels=['cifar10', 'svhn']):    
    
    fig = plt.figure(figsize=(12, 9))
    
    in_dist_Grads = args[0]
    out_dist_Grads = args[1]
    combined = np.concatenate((in_dist_Grads, out_dist_Grads))
    label_1 = np.ones(len(in_dist_Grads))
    label_2 = np.zeros(len(out_dist_Grads))
    label = np.concatenate((label_1, label_2))
    fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
    #plot_roc_curve(fpr, tpr)
    rocauc = metrics.auc(fpr, tpr)
    title = f'In-dist : {labels[0]}  /  Out-dist : {labels[1]} \n AUC for Gradient Norm is: {rocauc:.6f}'
    plt.plot(fpr, tpr)
    plt.title(title)
    plt.grid(True)
    plt.show()
    
    return rocauc