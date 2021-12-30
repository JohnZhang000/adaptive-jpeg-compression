# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:47:03 2021

@author: DELL
"""

import numpy as np
import torch
import os 
import sys
import torch.nn as nn
# import torch.nn.functional as F

from art.attacks.evasion import FastGradientMethod,DeepFool
from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
# from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin
from art.defences.postprocessor import ClassLabels,GaussianNoise,HighConfidence,ReverseSigmoid,Rounded
from scipy.special import softmax

from defense import defend_webpf_wrap,defend_webpf_my_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
from defense import defend_my_webpf
from defense_ago import defend_FD_ago_warp,defend_my_fd_ago
from adaptivce_defense import adaptive_defender

from models.cifar.allconv import AllConvNet
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from torch.utils.data import DataLoader

import json
sys.path.append('../common_code')
# from load_cifar_data import load_CIFAR_batch,load_CIFAR_train
import general as g
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train,load_imagenet_batch,load_imagenet_filenames
import pickle
from tqdm import tqdm
import logging

def append_attack(attacks,attack,model,epss):
    for i in range(len(epss)):
        attacks.append(attack(estimator=model,eps=epss[i]))   
        
        
def get_acc(fmodel,images,labels):
    predictions = fmodel.predict(images)
    predictions = np.argmax(predictions,axis=1)
    cors = np.sum(predictions==labels)
    return cors

def get_defended_acc(fmodel,dataloader,defenders):
    cors=np.zeros(defenders)
    for idx, (images, labels) in enumerate(dataloader):
        for idx_def,defender in enumerate(defenders):
            images_def,labels_def = defender(images.transpose(0,2,3,1).copy(),labels.copy())
            predictions = fmodel.predict(images_def)
            predictions = np.argmax(predictions,axis=1)
            cors[idx_def] += np.sum(predictions==labels)
    return cors

def get_defended_attacked_acc(fmodel,dataloader,attackers,defenders,defender_names):
    cors=np.zeros((len(attackers)+1,len(defenders)+1))
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        images=images.numpy()
        labels=labels.numpy()
        for j in range(len(attackers)+1):
            images_att=images.copy()
            eps=0
            if j>0:
                try:
                    eps=attackers[j-1].eps
                except:
                    eps=0
                images_att  = attackers[j-1].generate(x=images.copy())
            for k in range(len(defenders)+1):
                    images_def = images_att.copy()
                    if k>0:
                        if 'ADAD-flip'==defender_names[k-1]:
                            images_def,_ = defenders[k-1](images_att.transpose(0,2,3,1).copy(),labels.copy(),None,0)
                        elif 'ADAD+eps-flip'==defender_names[k-1]:
                            images_def,_ = defenders[k-1](images_att.transpose(0,2,3,1).copy(),labels.copy(),eps*np.ones(images_att.shape[0]),0)
                        else:
                            images_def,_ = defenders[k-1](images_att.transpose(0,2,3,1).copy(),labels.copy())
                        images_def=images_def.transpose(0,3,1,2)
                    cors[j,k] += get_acc(fmodel,images_def,labels)
    cors=cors/len(dataloader.dataset)
    return cors

if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=4:
        print('Manual Mode !!!')
        thresh0  = 0.00075
        thresh1  = 0.00075
        thresh2  = 0.00075
    else:
        print('Terminal Mode !!!')
        thresh0  = float(sys.argv[1])
        thresh1  = float(sys.argv[2])
        thresh2  = float(sys.argv[3])
    
    threshs=[thresh0,thresh1,thresh2]
    model_vanilla_type    = 'allconv'
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    logger=logging.getLogger(name='r')
    logger.setLevel(logging.FATAL)
    formatter=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    fh=logging.FileHandler(os.path.join(saved_dir,'acc_log.txt'))
    fh.setLevel(logging.FATAL)
    fh.setFormatter(formatter)
    
    ch=logging.StreamHandler()
    ch.setLevel(logging.FATAL)
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.fatal(('\n----------label record-----------'))
    
    '''
    加载cifar-10图像
    '''
    g.setup_seed(0)
    if 'imagenet' in model_vanilla_type:
        dataset_name='imagenet'
    else:
        dataset_name='cifar-10'
    data_setting=g.dataset_setting(dataset_name)
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,'val')
    dataloader = DataLoader(dataset, batch_size=data_setting.pred_batch_size, drop_last=False)    

    '''
    加载模型
    '''
    dir_model  = '../models/cifar_vanilla_'+model_vanilla_type+'.pth.tar'
    model,_=g.select_model(model_vanilla_type, dir_model)
    model.eval()
    
    fmodel = PyTorchClassifier(model = model,nb_classes=data_setting.nb_classes,clip_values=(0,1),
                               input_shape=data_setting.input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(data_setting.mean, data_setting.std))
   
    '''
    防御初始化
    '''

    defences_pre=[]
    defences_names_pre=[]
    # defences_pre.append(JpegCompression(clip_values=(0,1),quality=25,channels_first=False))
    # defences_names_pre.append('JPEG')
    # defences_pre.append(GaussianAugmentation(sigma=0.01,augmentation=False))
    # defences_names_pre.append('GauA')
    # defences_pre.append(SpatialSmoothing())
    # defences_names_pre.append('BDR')
    # defences_pre.append(defend_webpf_wrap)
    # defences_names_pre.append('webpf')
    # defences_pre.append(defend_rdg_wrap)
    # defences_names_pre.append('rdg')
    # defences_pre.append(defend_fd_wrap)
    # defences_names_pre.append('fd')
    # defences_pre.append(defend_shield_wrap)
    # defences_names_pre.append('shield')
    # defences_pre.append(defend_FD_ago_warp)
    # defences_names_pre.append('FD_ago')
    
    table_pkl=os.path.join(saved_dir,'table_dict.pkl')
    gc_model_dir='../saved_tests/img_attack_reg/spectrum_label/allconv/model_best.pth.tar'
    model_mean_std='../saved_tests/img_attack_reg/spectrum_label/allconv/mean_std_train.npy'
    # threshs=[0.001,0.001,0.001]
    # fd_ago_new=defend_my_fd_ago(table_pkl,gc_model_dir,[0.3,0.8,0.8],[0.0001,0.0001,0.0001],model_mean_std)
    # fd_ago_new.get_cln_dct(images.transpose(0,2,3,1).copy())
    # print(fd_ago_new.abs_threshs)
    # defences_pre.append(fd_ago_new.defend)
    # defences_names_pre.append('fd_ago_my')
    # defences_pre.append(fd_ago_new.defend_channel_wise_with_eps)
    # defences_names_pre.append('fd_ago_my')
    # defences_pre.append(fd_ago_new.defend_channel_wise)
    # defences_names_pre.append('fd_ago_my_no_eps')
    # defences_pre.append(fd_ago_new.defend_channel_wise_adaptive_table)
    # defences_names_pre.append('fd_ago_my_ada')
    adaptive_defender=adaptive_defender(table_pkl,gc_model_dir,model_mean_std)
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD')
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD-flip')
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD+eps-flip')
    
    
    '''
    攻击初始化
    '''
    attacks=[]
    attack_names=[]
    eps_L2=[0.1,0.5,1.0,10.0]
    eps_Linf=[0.005,0.01,0.1,1.0,10.0]
    
    for i in range(len(eps_L2)):
          attacks.append(FastGradientMethod(estimator=fmodel,eps=eps_L2[i],norm=2,eps_step=eps_L2[i]))
          attack_names.append('FGSM_L2_'+str(eps_L2[i]))    
    # for i in range(len(eps_L2)):
    #       attacks.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_L2[i],norm=2,batch_size=512,verbose=False))
    #       attack_names.append('PGD_L2_'+str(eps_L2[i]))    
    # attacks.append(DeepFool(classifier=fmodel,batch_size=512,verbose=False))
    # attack_names.append('DeepFool_L2')    
    # attacks.append(CarliniL2Method(classifier=fmodel,batch_size=512,verbose=False))
    # attack_names.append('CW_L2')
        

    '''
    计算防御效果
    '''            
    # 标为原始样本
 
    accs=get_defended_attacked_acc(fmodel,dataloader,attacks,defences_pre,defences_names_pre)
    np.save(os.path.join(saved_dir,'acc.npy'),accs)
    logger.fatal(attack_names)
    logger.fatal(defences_names_pre)
    logger.fatal(accs)
    logger.fatal(accs.mean(axis=0))
   