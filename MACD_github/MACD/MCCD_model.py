
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import torch.nn as nn
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from pytorch_revgrad import RevGrad
from MACD.utils import *
import torch
torch.autograd.set_detect_anomaly(True)




class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        in_dim, h_dim,out_dim=dim
        self.layer = nn.Sequential(nn.Linear(in_dim, h_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(h_dim),
                                   nn.Linear(h_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(out_dim),
                                   )
        self.init_weights()

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        out = self.layer(x.cuda())
        return out


class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        in_dim, h_dim, out_dim = dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(out_dim),
                                   )
        self.init_weights()

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        out = self.layer(x)
        return out

class Predictor(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(Predictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, out_dim),
            nn.Softmax(dim=1)
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.layer(x)
class Discriminator(nn.Module):
    def __init__(self,dim):
        super(Discriminator, self).__init__()
        in_dim, h_dim,out_dim=dim
        self.layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(h_dim),

            nn.Linear(h_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(out_dim),
            nn.Sigmoid()
                                   )
        self.init_weights()



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.layer(x)
        return out


class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        in_dim, h_dim, out_dim = dim
        self.layer = nn.Sequential(nn.Linear(in_dim, h_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(h_dim),
                                   nn.Linear(h_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.LayerNorm(out_dim),
                                   nn.Sigmoid()
                                   )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.layer(x)
        return out
class MDCD(nn.Module):
    def __init__(self, celltype_num, outdirfile, used_features,num_epochs):
        super(MDCD, self).__init__()
        self.num_epochs_new =num_epochs
        self.batch_size = 2048
        self.target_type = "real"
        self.learning_rate = 0.01
        self.celltype_num = celltype_num
        self.labels = None
        self.used_features = used_features
        # self.seed = 2021,150,1700,3000
        self.seed = 2021
        self.outdir = outdirfile
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        feature_num = len(self.used_features)
        dim=[feature_num, 1024,512]
        self.encoder_da = Encoder(dim).cuda()
        dim1=[512,1024,feature_num]
        self.decoder_da = Decoder(dim1).cuda()
        self.predictor_da =Predictor(512,celltype_num).cuda()
        dim2=[256,64,1]
        self.Discriminator = Discriminator(dim2).cuda()
        dim3=[256,128,1]
        self.Classifier = Classifier(dim3).cuda()
    def forward(self, x, lamda=1):
        self.revgrad = RevGrad(lamda).cuda()
        x = x.cuda()
        embedding_source = self.encoder_da(x)
        con_source = self.decoder_da(embedding_source)
        pro = self.predictor_da(embedding_source)
        znoise = embedding_source[:, :256]
        zbio = embedding_source[:, 256:]
        clas_out = self.Classifier(zbio)
        disc_out = self.Discriminator(self.revgrad(znoise))
        return embedding_source, con_source, pro, clas_out, disc_out

    def prepare_dataloader(self, sm_data, sm_label, st_data, batch_size):
        self.source_data_x = sm_data.values.astype(np.float32)
        self.source_data_y = sm_label.values.astype(np.float32)
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)
        self.used_features = list(sm_data.columns)
        self.target_data_x = torch.from_numpy(st_data.values.astype(np.float32))
        if self.target_type == "real":
            target_ratios = self.target_data_y = np.random.rand(st_data.shape[0], sm_label.shape[1])
            self.target_data_y = np.array(target_ratios, dtype=np.float32)
        else:
            print("target_type类型错误")
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False)

    def mask_features(self, X, mask_ratio):
        if isinstance(X, np.ndarray):
            mask = np.random.choice([True, False], size=X.shape, p=[mask_ratio, 1 - mask_ratio])
        elif isinstance(X, torch.Tensor):
            mask = torch.rand(X.shape) < mask_ratio
        else:
            raise TypeError("type error!")
        use_x = X.clone()  # 使用 X 的副本以避免改变原始数据
        use_x[mask] = 0
        return use_x, mask

    def prediction(self):
        self.eval()
        preds = None
        for batch_idx, (x, y) in enumerate(self.test_target_loader):
            x = x.cuda()
            embedding_source, con_source, pro, clas_out, disc_out = self.forward(x)
            logits = pro.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
        target_preds = pd.DataFrame(preds, columns=self.labels)
        model_and_settings = {
            'model': self.state_dict(),
            'seed': self.seed
        }
        torch.save(model_and_settings, self.outdir + '/model_with_settings.pth')
        return target_preds

    def double_train(self, sm_data, sm_label, st_data):
        self.train()
        self.prepare_dataloader(sm_data, sm_label, st_data, self.batch_size)
        self.optim = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.decoder_da.parameters()},], lr=self.learning_rate)
        self.optim1 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.predictor_da.parameters()},], lr=self.learning_rate)
        self.optim_discriminator = torch.optim.Adam(self.Discriminator.parameters(), lr=0.005)  # 判别器的学习率
        self.optim_classifier = torch.optim.Adam(self.Classifier.parameters(), lr=0.01)  # 分类器的学习率
        criterion_da = nn.MSELoss().cuda()
        metric_logger = defaultdict(list)
        epsilon=0.01
        for epoch in range(self.num_epochs_new):
            train_target_iterator = iter(self.train_target_loader)
            pred_loss_epoch, con_loss_epoch = 0., 0.
            dis_loss_epoch_y, dis_loss_epoch = 0.0, 0.0
            class_loss_epoch, class_loss_epoch_y = 0.0, 0.0
            all_loss_epoch=0.0
            for i in range(1):
                for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                    try:
                        target_x, _ = next(train_target_iterator)
                    except StopIteration:
                        train_target_iterator = iter(self.train_target_loader)
                        target_x, _ = next(train_target_iterator)

                    use_x, mask = self.mask_features(source_x.cuda(), 0.3)
                    source_x = source_x.cuda()  # 将source_x移动到CUDA设备

                    embedding_source, con_source, pro, clas_out, disc_out = self.forward(source_x * (~mask.cuda()))
                    embedding_source_y, con_source_y, pro_y, clas_out_y, disc_out_y = self.forward(target_x.cuda())

                    con_loss = criterion_da(source_x * (~mask.cuda()), con_source * (~mask.cuda()))
                    con_loss_epoch += con_loss.data.item()


                    source_label = torch.ones(disc_out.shape[0]).unsqueeze(1).cuda()  # 定义source domain label为1
                    source_label1 = source_label * (1 - epsilon) + (epsilon / 2)

                    target_label_y = torch.zeros(clas_out_y.shape[0]).unsqueeze(1).cuda()  # 定义target domain label为0
                    target_label_y1 = target_label_y * (1 - epsilon) + (epsilon / 2)
                    clas_loss = nn.BCELoss()(clas_out, source_label1)
                    dis_loss = nn.BCELoss()(disc_out, source_label1)
                    clas_loss_y = nn.BCELoss()(clas_out_y, target_label_y1)
                    dis_loss_y = nn.BCELoss()(disc_out_y, target_label_y1)

                    dis_loss_epoch += dis_loss.data.item()
                    dis_loss_epoch_y += dis_loss_y.data.item()
                    class_loss_epoch += clas_loss.data.item()
                    class_loss_epoch_y += clas_loss_y.data.item()
                    # loss = 10 * con_loss + dis_loss + 1000 * dis_loss_y + clas_loss + 1000 * clas_loss_y
                    loss = con_loss  + (dis_loss + dis_loss_y +clas_loss +clas_loss_y)
                    all_loss_epoch += loss.data.item()
                    self.optim.zero_grad()
                    self.optim_discriminator.zero_grad()
                    self.optim_classifier.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.optim_discriminator.step()
                    self.optim_classifier.step()
                    torch.cuda.empty_cache()


            for i in range(1):
                for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                    try:
                        target_x, _ = next(train_target_iterator)
                    except StopIteration:
                        train_target_iterator = iter(self.train_target_loader)
                        target_x, _ = next(train_target_iterator)

                    source_x = source_x.cuda()  # 将source_x移动到CUDA设备

                    embedding_source, con_source, pro, clas_out, disc_out = self.forward(source_x)
                    pred_loss = criterion_da(source_y.cuda(), pro)
                    pred_loss_epoch += pred_loss.data.item()
                    # loss1=1000*pred_loss
                    loss1 = pred_loss
                    self.optim1.zero_grad()
                    loss1.backward()
                    self.optim1.step()
                    torch.cuda.empty_cache()

            pred_loss_epoch = pred_loss_epoch / (batch_idx + 1)
            con_loss_epoch = con_loss_epoch / (batch_idx + 1)
            dis_loss_epoch_y = dis_loss_epoch_y / (batch_idx + 1)
            dis_loss_epoch = dis_loss_epoch / (batch_idx + 1)
            all_loss_epoch = all_loss_epoch / (batch_idx + 1)
            class_loss_epoch = class_loss_epoch / (batch_idx + 1)
            class_loss_epoch_y = class_loss_epoch_y / (batch_idx + 1)
            if epoch>0:
                metric_logger['pre_loss'].append(pred_loss_epoch)


                metric_logger['con_loss'].append(con_loss_epoch)


                metric_logger['dis_loss_y'].append(dis_loss_epoch_y)


                metric_logger['dis_loss'].append(dis_loss_epoch)

                metric_logger['all_loss'].append(all_loss_epoch)


                metric_logger['class_loss'].append(class_loss_epoch)


                metric_logger['class_loss_y'].append(class_loss_epoch_y)


            if (epoch+1) % 50== 0:
                print(
                    '============= Epoch {:02d}/{:02d} in stage ============='.format(epoch + 1, self.num_epochs_new))
                print(
                    "pre_loss=%f, con_loss=%f, dis_loss_y=%f,dis_loss=%f,class_loss_y=%f, class_loss=%f,total_loss_DA=%f" % (
                        pred_loss_epoch, con_loss_epoch, dis_loss_epoch_y, dis_loss_epoch, class_loss_epoch_y,
                        class_loss_epoch, all_loss_epoch))

        if self.target_type == "simulated":
            SaveLossPlot(self.outdir, metric_logger,
                         loss_type=['pred_loss', 'disc_loss', 'disc_loss_DA', 'target_ccc', 'target_rmse',
                                    'target_corr'], output_prex='Loss_metric_plot_stage3')
        elif self.target_type == "real":
            SaveLossPlot(self.outdir, metric_logger,
                         loss_type=['pre_loss', 'con_loss', 'dis_loss_y', 'dis_loss', 'class_loss_y', 'class_loss',
                                    'all_loss'],
                         output_prex='Loss_metric_plot_stage')

