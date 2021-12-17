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
import os
import time
import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle
import joblib
import logging
sys.path.append('../common_code')
import general as g

class Net(nn.Module):
    def __init__(self,input_size):
        super(Net,self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=10,kernel_size=3,stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3,stride=1)
        self.conv2 = nn.Conv1d(10, 20, 3, 1)
        self.max_pool2 = nn.MaxPool1d(3, 1)
        self.conv3 = nn.Conv1d(20, 40, 3, 1)
        
        self.linear1 = nn.Linear(40*6,120)
        self.linear2 = nn.Linear(120,84)
        self.linear3 = nn.Linear(84,2)
    
    def forward(self,x):
        x = x.view(-1,1,self.input_size)
        x = F.leaky_relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.leaky_relu(self.conv3(x))
        
        x = x.view(-1, 40*6)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv1d):
                if m.weight is not None:
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight.data,1.0)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,0.0)

class Net_imagenet(nn.Module):
    def __init__(self,input_size):
        super(Net_imagenet,self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=10,kernel_size=3,stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3,stride=1)
        self.conv2 = nn.Conv1d(10, 20, 3, 1)
        self.max_pool2 = nn.MaxPool1d(3, 1)
        self.conv3 = nn.Conv1d(20, 40, 3, 1)
        
        self.linear1 = nn.Linear(4080,120)
        self.linear2 = nn.Linear(120,84)
        self.linear3 = nn.Linear(84,2)
    
    def forward(self,x):
        x = x.view(-1,1,self.input_size)
        x = F.leaky_relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.leaky_relu(self.conv3(x))
        
        x = x.view(-1, 4080)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv1d):
                if m.weight is not None:
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight.data,1.0)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,0.0)

class spectrum_dataset(Dataset):
    def __init__(self,data_dir,label_dir,mean_std=None):
        self.spectrums=np.load(data_dir).astype(np.float32)
        self.labels=np.load(label_dir).astype(np.float32)
        self.mean_std=mean_std
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        spectrum=torch.from_numpy(self.spectrums[idx,:])
        if self.mean_std:
            spectrum = (spectrum-self.mean_std[0])/self.mean_std[1]
            
        label= int(self.labels[idx,:])
        return spectrum, label
    
def topK_loss(criterion, pred_label, gt_label):
    loss_wise = criterion(pred_label, gt_label)
    loss_sorted = loss_wise/loss_wise.sum()
    loss_sorted = loss_sorted.sort(descending=True)
    
    ratio=0.0
    break_point=0
    for i,v in enumerate(loss_sorted[0]):
        break_point=i
        if ratio>=0.75:
            break
        ratio+=v.data.detach().cpu().numpy()
    need_bp=loss_sorted[1][:break_point]
    loss_topk=loss_wise[need_bp].mean()
    return loss_topk

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

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     # random.seed(seed)
#     torch.backends.cudnn.deterministic = True

if __name__=='__main__':   
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=7:
        print('Manual Mode !!!')
        s_dir      = '../saved_tests/img_attack/spectrum_label'
        model_type    = 'allconv'
        device     = 3
        flag_manual_mode = 1
    else:
        print('Terminal Mode !!!')
        s_dir      = sys.argv[1]
        model_type    = sys.argv[2]
        device     = int(sys.argv[3])
        # max_lr     = float(sys.argv[1])
        # epochs     = int(sys.argv[2])
        # batch_size = int(sys.argv[3])
        # s_dir      = sys.argv[4]
        # vanilla_model = sys.argv[5]
        # device     = int(sys.argv[6])   
        flag_manual_mode = 0

    os.environ['CUDA_VISIBLE_DEVICES']=str(device)
    g.setup_seed(0)
    dataset       = 'cifar-10'
    if 'imagenet' in model_type:
        dataset='imagenet'
    data_dir      = os.path.join(s_dir,model_type)
    train_dataset = spectrum_dataset(data_dir+'/spectrums_train.npy',data_dir+'/labels_train.npy')
    test_dataset  = spectrum_dataset(data_dir+'/spectrums_test.npy',data_dir+'/labels_test.npy')        
    logging.basicConfig(filename=os.path.join(data_dir,'log_train.txt'),
                        level=logging.INFO)
    logging.info(('\n----------record-start-----------'))
    
    #adaboost params
    adb_max_depth=g.adb_max_depth
    adb_epochs=g.adb_epochs
    
    #svm params
    svm_gamma=g.svm_gamma
    svm_c=g.svm_c
    
    #cnn-params
    cnn_max_lr     = g.cnn_max_lr
    cnn_epochs     = g.cnn_epochs
    cnn_batch_size = g.cnn_batch_size
    
    '''
    Adaboost
    '''
    start_time = time.time()
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=adb_max_depth),
                             algorithm="SAMME",
                             n_estimators=adb_epochs)
    bdt.fit(train_dataset.spectrums,train_dataset.labels)
    
    joblib.dump(bdt,os.path.join(data_dir,model_type+'_adaboost.pkl'))
    clf_bdt=joblib.load(os.path.join(data_dir,model_type+'_adaboost.pkl'))
    end_time=time.time()
    logging.info(("[Adaboost] Train Time %f s") % (end_time-start_time))
    logging.info(("[Adaboost] Train acc %f") % clf_bdt.score(train_dataset.spectrums,train_dataset.labels))
    logging.info(("[Adaboost] Test acc %f") % clf_bdt.score(test_dataset.spectrums,test_dataset.labels))  
    print('Adaboost Done')

    
    '''
    SVM
    '''
    start_time = time.time()
    svc_p=SVC(kernel='rbf',gamma=svm_gamma,C=svm_c)
    svc_p.fit(train_dataset.spectrums,train_dataset.labels)
    joblib.dump(svc_p,os.path.join(data_dir,model_type+'_svm.pkl'))
    clf_svc=joblib.load(os.path.join(data_dir,model_type+'_svm.pkl'))
    end_time=time.time()
    logging.info(("[SVM] Train Time %f s") % (end_time-start_time))
    logging.info(("[SVM] Train acc %f") % clf_svc.score(train_dataset.spectrums,train_dataset.labels))
    logging.info(("[SVM] Test acc %f") % clf_bdt.score(test_dataset.spectrums,test_dataset.labels))
    print('SVM Done')
        
    '''
    CNN
    '''
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=cnn_batch_size)    
    test_loader   = DataLoader(test_dataset, shuffle=False, batch_size=cnn_batch_size)
    
    if 'cifar-10'==dataset:
        model     = Net(16).cuda()
    elif 'imagenet'==dataset:
        model     = Net_imagenet(112).cuda()
    model.init_weights()
    criterion = nn.CrossEntropyLoss()#reduce=(False)
    optimizer = torch.optim.Adam(model.parameters(),lr=cnn_max_lr)
    
    loss_list  = []
    acc_list   = []
    best_acc   = 0
    start_time = time.time()
    for epoch in range(cnn_epochs):
        
        train(train_loader,model,criterion,optimizer,epoch,loss_list)
        
        acc = test(test_loader, model, criterion)
        acc_list.append(acc)
    
        is_best  = acc > best_acc
        best_acc = max(acc,best_acc)
        
        save_checkpoint({
            'epoch':epoch+1,
            'state_dict':model.state_dict(),
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict(),
            }, is_best, data_dir)
    end_time=time.time()
    logging.info(("[CNN] Train Time %f s") % (end_time-start_time))
    logging.info(("[CNN] Train acc %f") % best_acc)

    
    if flag_manual_mode:       
        plt.figure()
        plt.plot(np.arange(0,len(loss_list)),loss_list)
        plt.title(str(best_acc))
        plt.figure()
        plt.plot(np.arange(0,len(loss_list)),acc_list)
        plt.title(str(best_acc))
        plt.show()        
    
    # 标记预测错误的数据
    label_gt=[]
    label_pred=[]
    data_err=[]
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
            
            idx_err=(predicted != y)
            data_err.append(X[idx_err,:].detach().cpu().numpy())
            label_gt.append(y[idx_err].detach().cpu().numpy())
            label_pred.append(predicted[idx_err].detach().cpu().numpy())
    acc=(correct/total)
    # print('【Test】 %.2f %% ' % (acc))
    logging.info(("[CNN] Test acc %f") % acc)
    print('CNN Done')
    
    label_gt_np=np.vstack((label_gt[i].reshape(-1,1) for i in range(len(label_gt))))
    label_pred_np=np.vstack((label_pred[i].reshape(-1,1) for i in range(len(label_pred))))
    labels_all=np.hstack([label_gt_np,label_pred_np])
    data_np=np.vstack((data_err[i].reshape(-1,16) for i in range(len(data_err))))
