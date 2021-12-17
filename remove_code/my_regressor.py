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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
import pickle
import joblib
import logging
sys.path.append('../common_code')
import general as g

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
            
        label= float(self.labels[idx,:])
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
    if len(sys.argv)!=2:
        print('Manual Mode !!!')  
        model_type    = 'allconv'
        # device     = 3
        flag_manual_mode = 1
    else:
        print('Terminal Mode !!!')
        model_type    = sys.argv[1]
        # device     = int(sys.argv[3])
        # max_lr     = float(sys.argv[1])
        # epochs     = int(sys.argv[2])
        # batch_size = int(sys.argv[3])
        # s_dir      = sys.argv[4]
        # vanilla_model = sys.argv[5]
        # device     = int(sys.argv[6])   
        flag_manual_mode = 0

    # os.environ['CUDA_VISIBLE_DEVICES']=str(1)
    g.setup_seed(0)
    dataset       = 'cifar-10'
    if 'imagenet' in model_type:
        dataset='imagenet'
    s_dir      = '../saved_tests/img_attack_reg/spectrum_label'
    data_dir      = os.path.join(s_dir,model_type)
    train_dataset = spectrum_dataset(data_dir+'/spectrums_test.npy',data_dir+'/labels_test.npy')
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
    bdt = AdaBoostRegressor(DecisionTreeRegressor(max_depth=adb_max_depth),
                              loss="square",
                              n_estimators=adb_epochs)
    bdt.fit(train_dataset.spectrums,train_dataset.labels)
    
    joblib.dump(bdt,os.path.join(data_dir,model_type+'_adaboost.pkl'))
    clf_bdt=joblib.load(os.path.join(data_dir,model_type+'_adaboost.pkl'))
    end_time=time.time()
    prt_info=("[Adaboost] Train Time %f s") % (end_time-start_time)
    print(prt_info)
    logging.info(prt_info)
    prt_info=("[Adaboost] Train acc %f") % clf_bdt.score(train_dataset.spectrums,train_dataset.labels)
    print(prt_info)
    logging.info(prt_info)
    prt_info=("[Adaboost] Test acc %f") % clf_bdt.score(test_dataset.spectrums,test_dataset.labels)
    print(prt_info)
    logging.info(prt_info)
    print('Adaboost Done')

    
    # '''
    # SVM
    # '''
    # start_time = time.time()
    # svc_p=SVR(kernel='rbf',gamma=svm_gamma,C=svm_c)
    # svc_p.fit(train_dataset.spectrums,train_dataset.labels)
    # joblib.dump(svc_p,os.path.join(data_dir,model_type+'_svm.pkl'))
    # clf_svc=joblib.load(os.path.join(data_dir,model_type+'_svm.pkl'))
    # end_time=time.time()
    # prt_info=("[SVM] Train Time %f s") % (end_time-start_time)
    # print(prt_info)
    # logging.info(prt_info)
    # prt_info=("[SVM] Train acc %f") % clf_svc.score(train_dataset.spectrums,train_dataset.labels)
    # print(prt_info)
    # logging.info(prt_info)
    # prt_info=("[SVM] Test acc %f") % clf_svc.score(test_dataset.spectrums,test_dataset.labels)
    # print(prt_info)
    # logging.info(prt_info)
    # print('Adaboost Done')
    
    '''
    other
    '''
    start_time = time.time()
    svc_p=LassoCV(normalize=True,cv=5,random_state=0)
    svc_p.fit(train_dataset.spectrums,train_dataset.labels)
    joblib.dump(svc_p,os.path.join(data_dir,model_type+'_other.pkl'))
    clf_svc=joblib.load(os.path.join(data_dir,model_type+'_other.pkl'))
    end_time=time.time()
    prt_info=("[other] Train Time %f s") % (end_time-start_time)
    print(prt_info)
    logging.info(prt_info)
    prt_info=("[other] Train acc %f") % clf_svc.score(train_dataset.spectrums,train_dataset.labels)
    print(prt_info)
    logging.info(prt_info)
    prt_info=("[other] Test acc %f") % clf_svc.score(test_dataset.spectrums,test_dataset.labels)
    print(prt_info)
    logging.info(prt_info)
    print('other Done')
    
    a=clf_svc.predict(test_dataset.spectrums)
    a1=test_dataset.labels
    aa=np.concatenate((a.reshape(-1,1),a1.reshape(-1,1)),axis=1)
    
        
