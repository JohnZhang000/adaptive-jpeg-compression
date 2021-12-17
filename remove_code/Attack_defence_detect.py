# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:47:03 2021

@author: DELL
"""

import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from my_spectrum_analyzer import img_spectrum_analyzer
import torch
# from torchvision import transforms #datasets, models, 
# from tqdm import tqdm
import os 
import sys
import torch.nn as nn
import torch.nn.functional as F

from models.cifar.allconv import AllConvNet
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from art.attacks.evasion import FastGradientMethod,DeepFool
from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
# from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from art.defences.detector.evasion import BinaryInputDetector, BinaryActivationDetector

import json

sys.path.append("..")
# from train_code.my_img_transformer import img_transformer
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.manifold import TSNE
sys.path.append('../common_code')
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Net_act(nn.Module):
    def __init__(self,input_size):
        super(Net_act,self).__init__()
        self.input_size = input_size
        self.conv1      = nn.Conv1d(in_channels=1,out_channels=10,kernel_size=3,stride=1)
        self.max_pool1  = nn.MaxPool1d(kernel_size=3,stride=1)
        self.flat1      = nn.Flatten()
        self.linear1    = nn.Linear(60,2)
    
    def forward(self,x):
        x = x.view(-1,1,self.input_size)
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.flat1(x)
        x = F.leaky_relu(self.linear1(x))      
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

class Net_bin(nn.Module):
    def __init__(self):
        super(Net_bin,self).__init__()
        self.conv1      = nn.Conv2d(3,4,5,padding=1)
        self.max_pool1  = nn.MaxPool2d(kernel_size=3,stride=1)
        self.flat1      = nn.Flatten()
        self.linear1    = nn.Linear(3136,2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.flat1(x)
        x = F.leaky_relu(self.linear1(x))      
        return x
    
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
                    
    
    
if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=4:
        print('Manual Mode !!!')
        model_type    = 'allconv'
        data          = 'test'
        device        = 2
    else:
        print('Terminal Mode !!!')
        model_type  = sys.argv[1]
        data        = sys.argv[2]
        device      = int(sys.argv[3])
        
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_type  
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)  
    '''
    加载cifar-10图像
    '''
    os.environ['CUDA_VISIBLE_DEVICES']=str(device)
    setup_seed(0)
    dir_cifar     = '../../../../../media/ubuntu204/F/Dataset/Dataset_tar/cifar-10-batches-py'
    # if 'test'==data:
    images_test,labels_test = load_CIFAR_batch(os.path.join(dir_cifar,'test_batch'))
    # elif 'train'==data:
    images_train,labels_train = load_CIFAR_train(dir_cifar)
    
    images_test=images_test.transpose(0,3,1,2)
    images_train=images_train.transpose(0,3,1,2)
    
    
    '''
    加载模型
    '''
    if model_type == 'allconv':
        model      = AllConvNet(10).eval()
        dir_model  = '../models/cifar_vanilla_allconv.pth.tar'
    elif model_type == 'densenet':
        model = densenet(num_classes=10).eval()
        dir_model  = '../models/cifar_vanilla_densenet.pth.tar'
    elif model_type == 'wrn':
        model = WideResNet(40, 10, 2, 0.0).eval()
        dir_model  = '../models/cifar_vanilla_wrn.pth.tar'
    elif model_type == 'resnext':
        model = resnext29(num_classes=10).eval()
        dir_model  = '../models/cifar_vanilla_resnext.pth.tar' 
    else:
        print('Error model name!!!')
    checkpoint = torch.load(dir_model)
    model      = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint["state_dict"],True) 
    with open(os.path.join("../models/cifar-10_class_to_idx.json")) as f:
        features=json.load(f)
    mean   = np.array((0.5,0.5,0.5),dtype=np.float32)
    std    = np.array((0.5,0.5,0.5),dtype=np.float32)
    fmodel = PyTorchClassifier(model = model,nb_classes=10,clip_values=(0,1),
                               input_shape=(3,32,32),loss = nn.CrossEntropyLoss(),
                               preprocessing=(mean, std))   
       
    
    '''
    攻击初始化
    '''
    attacks_train=[]
    attack_train_names=[]
    eps_L2=[0.1,0.5,1.0,10.0,100.0]
    eps_Linf=[0.005,0.01,0.1,1.0,10.0]
    for i in range(len(eps_L2)):
        attacks_train.append(FastGradientMethod(estimator=fmodel,eps=eps_L2[i],norm=2,eps_step=eps_L2[i]))
        attack_train_names.append('FGSM_L2_'+str(eps_L2[i]))
    for i in range(len(eps_L2)):
        attacks_train.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_L2[i],norm=2,batch_size=512,verbose=False))
        attack_train_names.append('PGD_L2_'+str(eps_L2[i]))      
    attacks_train.append(DeepFool(classifier=fmodel,batch_size=512,verbose=False))
    attack_train_names.append('DeepFool_L2')    
    attacks_train.append(CarliniL2Method(classifier=fmodel,batch_size=512,verbose=False))
    attack_train_names.append('CW_L2')
    
    for i in range(len(eps_Linf)):
        attacks_train.append(FastGradientMethod(estimator=fmodel,eps=eps_Linf[i],norm=np.inf,eps_step=eps_Linf[i]))
        attack_train_names.append('FGSM_Linf_'+str(eps_Linf[i]))    
    for i in range(len(eps_Linf)):
        attacks_train.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_Linf[i],norm=np.inf,batch_size=512,verbose=False))
        attack_train_names.append('PGD_Linf_'+str(eps_Linf[i]))     
    for i in range(len(eps_Linf)):
        attacks_train.append(CarliniLInfMethod(classifier=fmodel,batch_size=512,eps=eps_Linf[i],verbose=False))
        attack_train_names.append('CW_Linf_'+str(eps_Linf[i]))  
     
    # test    
    attacks=[]
    attack_names=[]    
    for i in range(len(eps_L2)):
          attacks.append(FastGradientMethod(estimator=fmodel,eps=eps_L2[i],norm=2,eps_step=eps_L2[i]))
          attack_names.append('FGSM_L2_'+str(eps_L2[i]))    
    for i in range(len(eps_L2)):
          attacks.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_L2[i],norm=2,batch_size=512,verbose=False))
          attack_names.append('PGD_L2_'+str(eps_L2[i]))    
    attacks.append(DeepFool(classifier=fmodel,batch_size=512,verbose=False))
    attack_names.append('DeepFool_L2')    
    attacks.append(CarliniL2Method(classifier=fmodel,batch_size=512,verbose=False))
    attack_names.append('CW_L2')
    
    
    for i in range(len(eps_Linf)):
         attacks.append(FastGradientMethod(estimator=fmodel,eps=eps_Linf[i],norm=np.inf,eps_step=eps_Linf[i]))
         attack_names.append('FGSM_Linf_'+str(eps_Linf[i]))    
    for i in range(len(eps_Linf)):
         attacks.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_Linf[i],norm=np.inf,batch_size=512,verbose=False))
         attack_names.append('PGD_Linf_'+str(eps_Linf[i]))     
    for i in range(len(eps_Linf)):
         attacks.append(CarliniLInfMethod(classifier=fmodel,batch_size=512,eps=eps_Linf[i],verbose=False))
         attack_names.append('CW_Linf_'+str(eps_Linf[i]))   
        
    
    '''
    标记数据
    '''  
    flag_load=0
    data_saved_dir='../saved_tests/img_attack/spectrum_label'
    
    if not flag_load:
        images_list=[images_train]
        labels_list=[np.array([[0,1]]*labels_train.shape[0])]       
        num_sel=int(np.round(images_train.shape[0]/len(attacks)))
        
        for i in range(len(attack_train_names)):
            idx_sel=np.random.permutation(images_train.shape[0])[0:num_sel]
            images_sel=images_train[idx_sel,:]
            labels_sel=labels_train[idx_sel]
            
            images_adv = attacks_train[i].generate(x=images_sel,y=labels_sel)
            images_list.append(images_adv)
            labels_list.append(np.array([[1,0]]*labels_sel.shape[0]))
            
        
        images_all=np.vstack(images_list)
        labels_all=np.vstack(labels_list)
        
        np.save(os.path.join(data_saved_dir,'defense_detect_images.npy'),images_all)
        np.save(os.path.join(data_saved_dir,'defense_detect_labels.npy'),labels_all)
    
    else:
        images_all = np.load(os.path.join(data_saved_dir,'defense_detect_images.npy'))
        labels_all = np.load(os.path.join(data_saved_dir,'defense_detect_labels.npy'))
    torch.cuda.empty_cache()    
    
    '''
    检测模型定义
    '''  
    model_det_bin = Net_bin()
    mean   = np.array((0.5,0.5,0.5),dtype=np.float32)
    std    = np.array((0.5,0.5,0.5),dtype=np.float32)
    fmodel_det_bin = PyTorchClassifier(
        model = model_det_bin,
        nb_classes=2,
        clip_values=(0,1),
        input_shape=(3,32,32),
        loss = nn.CrossEntropyLoss(),
        preprocessing=(mean, std),
        optimizer=torch.optim.Adam(model_det_bin.parameters(),lr=0.1),
        )
    
    activation_shape=fmodel.get_activations(images_all[0:2,:],0,batch_size=128).shape[1:][0]
    model_det_act = Net_act(activation_shape)
    fmodel_det_act = PyTorchClassifier(
        model = model_det_act,
        nb_classes=2,
        clip_values=(0,1),
        input_shape=activation_shape,
        loss = nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model_det_act.parameters(),lr=0.1),
        )

    '''
    检测模型训练
    '''  
    detectors = []
    detectors_names=[]
    fprint_list=[]
    
    detector1 = BinaryInputDetector(fmodel_det_bin)
    detector1.fit(images_all, labels_all, nb_epochs=1000, batch_size=128)
    detector1.save('det-bid',saved_dir)
    pred_all=detector1.predict(images_all)
    cor = np.sum(np.argmax(pred_all,axis=1)==np.argmax(labels_all,axis=1))
    prt_info='%s: %.1f'%('train_acc',100*cor/len(labels_all))
    print(prt_info)
    fprint_list.append(prt_info)
    detectors.append(detector1)
    detectors_names.append('Bin')
    
    torch.cuda.empty_cache()  
    
    detector2 = BinaryActivationDetector(classifier=fmodel,detector=fmodel_det_act,layer=0)
    detector2.fit(images_all, labels_all, nb_epochs=1000, batch_size=128)
    detector1.save('det-bad',saved_dir)
    pred_all=detector2.predict(images_all)
    cor = np.sum(np.argmax(pred_all,axis=1)==np.argmax(labels_all,axis=1))
    prt_info='%s: %.1f'%('train_acc',100*cor/len(labels_all))
    print(prt_info)
    fprint_list.append(prt_info)
    detectors.append(detector2)
    detectors_names.append('Act')
    torch.cuda.empty_cache()  
    

    '''
    防御效果测试
    '''  
    prt_info='\n------------------'
        
    file_log=os.path.join(saved_dir,'result_detect_log.txt')
    f=open(file_log,'w')
    f.write('result_detect\n')
    prt_info='\n------------------'
    print(prt_info)
    fprint_list.append(prt_info)
    labels=np.zeros_like(labels_test)
    for i in range(len(attacks)):
        images_now=images_test
        labels_now=labels_test            
        images_adv = attacks[i].generate(x=images_now,y=labels_now)
        for idx_det,detector in enumerate(detectors):
            labels=np.ones_like(labels_now)
            preds= detector.predict(images_adv)            
            cor_det = np.sum(np.argmax(preds,axis=1)==0)
            
            cor=cor_det
            
            prt_info='adv %s %s: %.1f'%(attack_names[i],detectors_names[idx_det],100*cor/len(labels))
            print(prt_info)
            fprint_list.append(prt_info)
            f=open(file_log,'a')
            f.write(prt_info+'\n')
            f.close()
        prt_info='\n'
        print(prt_info)
        fprint_list.append(prt_info)
        
        
