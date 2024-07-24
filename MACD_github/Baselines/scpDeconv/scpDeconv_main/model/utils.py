import os
from sklearn import preprocessing as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import scanpy as sc
import torch
import torch.nn as nn

### loss function ###
def L1_loss(preds, gt):
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss

def Recon_loss(recon_data, input_data):
    loss_rec_fn = nn.MSELoss().cuda()
    loss= loss_rec_fn(recon_data, input_data)
    return loss

### evaluate metrics ###
def ccc(preds, gt):
    numerator = 2 * np.corrcoef(gt, preds)[0][1] * np.std(gt) * np.std(preds)
    denominator = np.var(gt) + np.var(preds) + (np.mean(gt) - np.mean(preds)) ** 2
    ccc_value = numerator / denominator
    return ccc_value

def compute_metrics(preds, gt):
    gt = gt[preds.columns] # Align pred order and gt order  
    x = pd.melt(preds)['value']
    y = pd.melt(gt)['value']
    CCC = ccc(x, y)
    RMSE = sqrt(mean_squared_error(x, y))
    Corr = pearsonr(x, y)[0]
    return CCC, RMSE, Corr

def sample_normalize(data, normalize_method = 'min_max'):
    # Normalize data
    mm = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    if normalize_method == 'min_max':
        # it scales features so transpose is needed
        data = mm.fit_transform(data.T).T   
    elif normalize_method == 'z_score':
        # Z score normalization
        data = (data - data.mean(0))/(data.std(0)+(1e-10))
    return data

def SaveLossPlot(SavePath, metric_logger, loss_type, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    for i in range(len(loss_type)):
        plt.subplot(2, 3, i+1)
        plt.plot(metric_logger[loss_type[i]])
        plt.title(loss_type[i], x = 0.5, y = 0.5)
    imgName = os.path.join(SavePath, output_prex +'.png')
    plt.savefig(imgName)
    plt.close()

def SavePredPlot(SavePath, target_preds, ground_truth):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    celltypes = list(target_preds.columns)

    plt.figure(figsize=(5*(len(celltypes)+1), 5)) 

    eval_metric = []
    x = pd.melt(target_preds)['value']
    y = pd.melt(ground_truth)['value']
    eval_metric.append(ccc(x, y))
    eval_metric.append(sqrt(mean_squared_error(x, y)))
    eval_metric.append(pearsonr(x, y)[0])
    plt.subplot(1, len(celltypes)+1, 1)
    plt.xlim(0, max(y))
    plt.ylim(0, max(y))
    plt.scatter(x, y, s=2)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
    plt.text(0.05, max(y)-0.05, text, fontsize=8, verticalalignment='top')
    plt.title('All samples')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')

    for i in range(len(celltypes)):
        eval_metric = []
        x = target_preds[celltypes[i]]
        y = ground_truth[celltypes[i]]
        eval_metric.append(ccc(x, y))
        eval_metric.append(sqrt(mean_squared_error(x, y)))
        eval_metric.append(pearsonr(x, y)[0])
        plt.subplot(1, len(celltypes)+1, i+2)
        plt.xlim(0, max(y))
        plt.ylim(0, max(y))
        plt.scatter(x, y, s=2)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"r--")
        text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
        plt.text(0.05, max(y)-0.05, text, fontsize=8, verticalalignment='top')
        plt.title(celltypes[i])
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
    imgName = os.path.join(SavePath, 'pred_fraction_target_scatter.jpg')
    plt.savefig(imgName)
    plt.close()

def SavetSNEPlot(SavePath, ann_data, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    sc.tl.pca(ann_data, svd_solver='arpack')
    sc.pp.neighbors(ann_data, n_neighbors=10, n_pcs=20)
    sc.tl.tsne(ann_data)
    with plt.rc_context({'figure.figsize': (8, 8)}):
        sc.pl.tsne(ann_data, color=list(ann_data.obs.columns), color_map='viridis',frameon=False)
        plt.tight_layout()
    plt.savefig(os.path.join(SavePath, output_prex + "_tSNE_plot.jpg"))
    ann_data.write(os.path.join(SavePath, output_prex + ".h5ad"))
    plt.close()
    