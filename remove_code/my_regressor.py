# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:55:21 2021

@author: DELL
"""

import shutil
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss,MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD,Adam,lr_scheduler
import os
import time
import sys
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
import pickle
import joblib
import logging
from models.resnet_reg import resnet50,resnet18
from pytorchtools import EarlyStopping
sys.path.append('../common_code')
import general as g
torch.multiprocessing.set_sharing_strategy('file_system')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(10, 1)
        self.dp1 = nn.Dropout(0.25)
        self.dp2 = nn.Dropout(0.25)
        self.dp3 = nn.Dropout(0.25)

    def forward(self, x):
        # print(x.shape)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        # print(y.shape)
        y = y.contiguous().view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.dp1(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.dp2(y)
        y = self.fc3(y)
        y = self.relu5(y)
        y = self.dp3(y)
        y = self.fc4(y)
        # y = self.relu6(y)
        # y = self.fc5(y)
        y = y.squeeze(1)
        return y
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight.data,1.0)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,0.0)
                    
class spectrum_dataset(Dataset):
    def __init__(self,data_dir,mean_std=None):
        data=np.load(data_dir)
        self.spectrums=data['spectrums'].astype(np.float32).transpose(0,3,1,2)
        self.labels=data['labels'].astype(np.float32)
        self.mean_std=None
        if mean_std:
            self.mean_std=np.load(mean_std).astype(np.float32).transpose(2,0,1)
            self.spectrums=(self.spectrums-self.mean_std[0:3,...])/self.mean_std[3:6,...]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        spectrum=torch.from_numpy(self.spectrums[idx,:])
        # if None!=self.mean_std.any():
        #     mean_std=self.mean_std.transpose
        #     spectrum = (spectrum-self.mean_std[0:3,...])/self.mean_std[3:6,...]
            
        label= self.labels[idx]
        return spectrum, label

def train(train_loader, model, criterion, optimizer, epoch,loss_list):
    model.train()
    for batch,(X,Y) in enumerate(train_loader):
        
        X=X.cuda(non_blocking=True)
        Y=Y.cuda(non_blocking=True)
        optimizer.zero_grad()
        
        y_pred = model(X)
        
        loss = criterion(y_pred,Y)
        
        loss.backward()
        
        optimizer.step()
    loss_list.append(loss.data.item())
    print(("【Train】[Epoch] %3d [batch] %3d [loss] %f")%(epoch,batch,loss.data.item()))

def test(test_loader, model, criterion):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
    
            X, y = data 
            X=X.cuda(non_blocking=True)
            y=y.cuda(non_blocking=True)
            y_pred = model(X)
    
            _,predicted = torch.max(y_pred.data,dim=1)
    
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc=(correct/total)
    print('【Test】 %.2f %% ' % (100*acc))
    return acc

def save_checkpoint(state, is_best, s_dir, filename='checkpoint.pth.tar'):
    torch.save(state,os.path.join(s_dir,filename))
    if is_best:
        shutil.copyfile(os.path.join(s_dir,filename),os.path.join(s_dir,'model_best.pth.tar'))

def my_loss(pred,gt):
    # pred_in=pred
    # gt_in=gt
    # diff=(pred-gt)/gt_in
    # diff=torch.clamp_max(diff.abs(), 100)
    
    factor=1+torch.abs(torch.log10(gt/10))
    mse=pred-gt
    
    loss=torch.pow(factor,2)*torch.pow(mse,2)#(mse,2)
    
    return loss.mean()
    
if __name__=='__main__':   
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=2:
        print('Manual Mode !!!')  
        model_type    = 'vgg16_imagenet'
        # device     = 3
        flag_manual_mode = 1
    else:
        print('Terminal Mode !!!')
        model_type    = sys.argv[1]  
        flag_manual_mode = 0

    # os.environ['CUDA_VISIBLE_DEVICES']=str(1)
    g.setup_seed(0)
    dataset_name       = 'cifar-10'
    if 'imagenet' in model_type:
        dataset_name='imagenet'
    print(dataset_name)
    data_setting=g.dataset_setting(dataset_name)
    data_dir      = '../saved_tests/img_attack/'+model_type
    #cnn-params
    cnn_max_lr     = data_setting.cnn_max_lr
    cnn_epochs     = data_setting.cnn_epochs
    cnn_batch_size = data_setting.cnn_batch_size
    
    mean_std=data_dir+'/mean_std_train.npy'
    train_dataset = spectrum_dataset(data_dir+'/train.npy.npz',mean_std)
    test_dataset  = spectrum_dataset(data_dir+'/val.npy.npz',mean_std) 
    train_loader = DataLoader(train_dataset, batch_size=cnn_batch_size,shuffle=True,num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cnn_batch_size,num_workers=0, pin_memory=True)       

    
    logger=logging.getLogger(name='r')
    logger.setLevel(logging.FATAL)
    formatter=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    fh=logging.FileHandler(os.path.join(data_dir,'log_train.txt'))
    fh.setLevel(logging.FATAL)
    fh.setFormatter(formatter)
    
    ch=logging.StreamHandler()
    ch.setLevel(logging.FATAL)
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    
    # model = Net()
    model = resnet50(data_setting.nb_classes)
    model.init_weights()
    model = torch.nn.DataParallel(model).cuda()
    optimizer = Adam(model.parameters(),lr=cnn_max_lr,weight_decay=1e-4)
    # optimizer = SGD(model.parameters(),lr=cnn_max_lr, momentum=0.9,weight_decay=1e-4)
    # scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=cnn_max_lr,
    #                                     total_steps=int(cnn_epochs*len(train_loader)/data_setting.accum_grad_num), 
    #                                     verbose=False)
    cost = my_loss#MSELoss(reduction='mean')#CrossEntropyLoss()
    epoch = cnn_epochs
    best_loss = 100000000
    early_stoper=EarlyStopping(patience=10, verbose=False, delta=0, trace_func=logger.fatal)
    
    for _epoch in range(epoch):
        loss_prt=0
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            # label_np = np.zeros((train_label.shape[0], 10))
            train_x=train_x.cuda()
            train_label=train_label.cuda()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label)
            # loss = loss/data_setting.accum_grad_num
            loss_prt = loss.detach().cpu().numpy()
            loss.backward()
            if 0==(idx+1)%data_setting.accum_grad_num:
                if (idx+1) % (data_setting.train_print_epoch*data_setting.accum_grad_num) == 0:
                    lr_show=optimizer.state_dict()['param_groups'][0]['lr']#scheduler.get_lr()[0]
                    logger.fatal('[Epoch]:{}, idx: {}, loss: {}, lr: {}'.format(_epoch, idx, loss_prt,lr_show))
                clip_grad_norm_(model.parameters(),max_norm=10,norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
                model.zero_grad()
                loss_prt = 0
    
        correct = 0
        sum_loss = 0
        
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x=test_x.cuda()
                test_label=test_label.cuda()
                
                predict_y = model(test_x.float()).detach()
                label_np = test_label#.numpy()
                test_loss=cost(predict_y, label_np)
                # test_loss=test_loss/data_setting.accum_grad_num
                sum_loss+=test_loss.detach().cpu().numpy()
                # torch.cuda.empty_cache()

        ave_loss=sum_loss / (idx+1)    
        is_best = ave_loss<best_loss
        if is_best:
            best_loss=ave_loss
        
        logger.fatal('[Epoch]:{}, ave_loss: {:.8f}'.format(_epoch, ave_loss))
        # torch.save(model, 'models/mnist_{:.4f}.pkl'.format(ave_loss))
        save_checkpoint({'epoch':_epoch+1,
                         'state_dict':model.state_dict(),
                         'optimizer':optimizer.state_dict(),
                         }, is_best, data_dir)
        early_stoper(ave_loss)
        if early_stoper.early_stop:
            break
    
    # preds=[]
    # gts=[]
    # model.eval()
    # for idx, (test_x, test_label) in enumerate(test_loader):
    #     test_x=test_x.cuda()
    #     predict_y = model(test_x.float()).detach()
    #     preds.append(predict_y.detach().cpu().numpy())
    #     gts.append(test_label.numpy())
    
    # gts_np=np.hstack(gts)
    # preds_np=np.hstack(preds)
    # a=np.hstack((gts_np.reshape(-1,1),preds_np.reshape(-1,1)))
    # a1=a[:,1]-a[:,0]
    # a2=(a1)/a[:,0]
    # ave_loss=sum_loss / len(test_loader.dataset)
    # print('ave_loss: {:.4f}'.format(ave_loss))