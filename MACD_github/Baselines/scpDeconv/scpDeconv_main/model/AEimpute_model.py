import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from Baselines.scpDeconv.scpDeconv_main.model.utils import *

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        out = self.layer(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        out = self.layer(x)
        return out
      
class AEimpute(object):
    def __init__(self, option_list):
        self.num_epochs = 200
        self.batch_size = option_list['batch_size']
        self.learning_rate = option_list['learning_rate']
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021
        self.outdir = option_list['SaveResultsDir']

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def AEimpute_model(self, celltype_num):
        feature_num = len(self.used_features)

        self.encoder_im = nn.Sequential(EncoderBlock(feature_num, 512), 
                                        EncoderBlock(512, 256))

        self.predictor_im = nn.Sequential(nn.Linear(256, celltype_num), 
                                          nn.Softmax(dim=-1))

        self.decoder_im = nn.Sequential(DecoderBlock(256, 512), 
                                        DecoderBlock(512, feature_num))

        model_im = nn.ModuleList([])
        model_im.append(self.encoder_im)
        model_im.append(self.predictor_im)
        model_im.append(self.decoder_im)
        return model_im

    def prepare_dataloader(self, ref_data, target_data, batch_size):
        ### Prepare data loader for training ###
        # ref dataset
        ref_ratios = [ref_data.obs[ctype] for ctype in ref_data.uns['cell_types']]
        self.ref_data_x = ref_data.X.astype(np.float32)
        self.ref_data_y = np.array(ref_ratios, dtype=np.float32).transpose()

        tr_data = torch.FloatTensor(self.ref_data_x)
        tr_labels = torch.FloatTensor(self.ref_data_y)
        ref_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_ref_loader = Data.DataLoader(dataset=ref_dataset, batch_size=batch_size, shuffle=True)
        self.test_ref_loader = Data.DataLoader(dataset=ref_dataset, batch_size=batch_size, shuffle=False)

        # Extract celltype and feature info
        self.labels = ref_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(ref_data.var_names)

        # Target dataset
        self.target_data_x = target_data.X.astype(np.float32)
        self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    def train(self, ref_data, target_data):
        ### prepare model structure ###
        self.prepare_dataloader(ref_data, target_data, self.batch_size)
        self.model_im = self.AEimpute_model(self.celltype_num).cuda()

        ### setup optimizer ###
        optimizer_im = torch.optim.Adam([{'params': self.encoder_im.parameters()},
                                         {'params': self.predictor_im.parameters()},
                                         {'params': self.decoder_im.parameters()}], lr=self.learning_rate)

        metric_logger = defaultdict(list)

        for epoch in range(self.num_epochs):
            self.model_im.train()

            train_target_iterator = iter(self.train_target_loader)
            loss_epoch, pred_loss_epoch, recon_loss_epoch = 0., 0., 0.
            for batch_idx, (ref_x, ref_y) in enumerate(self.train_ref_loader):
                # get batch item of target
                try:
                    target_x, _ = next(train_target_iterator)
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)

                X = torch.cat((ref_x, target_x))

                embedding = self.encoder_im(X.cuda())
                frac_pred = self.predictor_im(embedding)
                recon_X = self.decoder_im(embedding)

                # caculate loss 
                pred_loss = L1_loss(frac_pred[range(self.batch_size),], ref_y.cuda()) 
                pred_loss_epoch += pred_loss
                rec_loss = Recon_loss(recon_X, X.cuda())
                recon_loss_epoch += rec_loss
                loss = rec_loss + pred_loss
                loss_epoch += loss   

                # update weights
                optimizer_im.zero_grad()
                loss.backward()
                optimizer_im.step()

            loss_epoch = loss_epoch/(batch_idx + 1)
            metric_logger['cAE_loss'].append(loss_epoch)
            pred_loss_epoch = pred_loss_epoch/(batch_idx + 1)
            metric_logger['pred_loss'].append(pred_loss_epoch)
            recon_loss_epoch = recon_loss_epoch/(batch_idx + 1)
            metric_logger['recon_loss'].append(recon_loss_epoch)
            if (epoch+1) % 10 == 0:
                print('============= Epoch {:02d}/{:02d} in stage2 ============='.format(epoch + 1, self.num_epochs))
                print("cAE_loss=%f, pred_loss=%f, recon_loss=%f" % (loss_epoch, pred_loss_epoch, recon_loss_epoch))

        ### Plot loss ###
        SaveLossPlot(self.outdir, metric_logger, loss_type = ['cAE_loss','pred_loss','recon_loss'], output_prex = 'Loss_plot_stage2')

        ### Save reconstruction data of ref and target ###
        ref_recon_data = self.write_recon(ref_data)

        return ref_recon_data

    def write_recon(self, ref_data):
        
        self.model_im.eval()
        
        ref_recon, ref_label = None, None
        for batch_idx, (x, y) in enumerate(self.test_ref_loader):
            x_embedding = self.encoder_im(x.cuda())
            x_prediction = self.predictor_im(x_embedding)
            x_recon = self.decoder_im(x_embedding).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
            ref_recon = x_recon if ref_recon is None else np.concatenate((ref_recon, x_recon), axis=0)
            ref_label = labels if ref_label is None else np.concatenate((ref_label, labels), axis=0)
        ref_recon = pd.DataFrame(ref_recon, columns=self.used_features)
        ref_label = pd.DataFrame(ref_label, columns=self.labels)

        ref_recon_data = ad.AnnData(X=ref_recon.to_numpy(), obs=ref_label)
        ref_recon_data.uns['cell_types'] = self.labels
        ref_recon_data.var_names = self.used_features

        ### Plot recon ref TSNE plot ###
        # SavetSNEPlot(self.outdir, ref_recon_data, output_prex='AE_Recon_ref')
        ### Plot recon ref TSNE plot using missing features ###
        # sc.pp.filter_genes(ref_data, min_cells=0)
        # missing_features = list(ref_data.var[ref_data.var['n_cells']==0].index)
        # if len(missing_features) > 0:
        #     Recon_ref_data_new = ref_recon_data[:,missing_features]
        #     SavetSNEPlot(self.outdir, Recon_ref_data_new, output_prex='AE_Recon_ref_missingfeature')

        return ref_recon_data
