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

from defense import defend_webpf_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
from defense_ago import defend_FD_ago_warp

from models.cifar.allconv import AllConvNet
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
import json
sys.path.append('../common_code')
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train

def append_attack(attacks,attack,model,epss):
    for i in range(len(epss)):
        attacks.append(attack(estimator=model,eps=epss[i]))   

if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=4:
        print('Manual Mode !!!')
        model_type    = 'allconv'
        data          = 'test'
        device        = 0
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
    dir_cifar     = '../../../../../media/ubuntu204/F/Dataset/Dataset_tar/cifar-10-batches-py'
    if 'test'==data:
        images,labels = load_CIFAR_batch(os.path.join(dir_cifar,'test_batch'))
    elif 'train'==data:
        images,labels = load_CIFAR_train(dir_cifar)
    else:
        print('Wrong data mode !!!')

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
    fmodel = PyTorchClassifier(
        model = model,
        nb_classes=10,
        clip_values=(0,1),
        input_shape=(3,32,32),
        loss = nn.CrossEntropyLoss(),
        preprocessing=(mean, std))
    
    pred_cln = fmodel.predict(images.transpose(0,3,1,2))
   
    '''
    防御初始化
    '''

    defences_pre=[]
    defences_names_pre=[]
    # defences_pre.append(JpegCompression(clip_values=(0,1),quality=25,channels_first=False))
    # defences_names_pre.append('JPEG')
#    defences_pre.append(FeatureSqueezing(clip_values=(0,1),bit_depth=32))
#    defences_names_pre.append('FeaS')
    # defences_pre.append(GaussianAugmentation(sigma=0.01,augmentation=False))
    # defences_names_pre.append('GauA')
    # defences_pre.append(LabelSmoothing(apply_predict=True))
    # defences_names_pre.append('LabS')
    # defences_pre.append(Resample(sr_original=16000,sr_new=8000,channels_first=False))
    # defences_names_pre.append('Resa')
    # defences_pre.append(SpatialSmoothing())
    # defences_names_pre.append('SpaS')
#    defences_pre.append(ThermometerEncoding(clip_values=(0,1),num_space=1))
#    defences_names_pre.append('TheE')
#    defences_pre.append(TotalVarMin())
#    defences_names_pre.append('ToVM')
    defences_pre.append(defend_webpf_wrap)
    defences_names_pre.append('webpf')
    defences_pre.append(defend_rdg_wrap)
    defences_names_pre.append('rdg')
    defences_pre.append(defend_fd_wrap)
    defences_names_pre.append('fd')
    defences_pre.append(defend_bdr_wrap)
    defences_names_pre.append('bdr')
    defences_pre.append(defend_shield_wrap)
    defences_names_pre.append('shield')
    defences_pre.append(defend_FD_ago_warp)
    defences_names_pre.append('FD_ago')
    
    
    
    defences_after=[]
    defences_names_aft=[]
#    defences_after.append(ClassLabels())
#    defences_names_aft.append('ClsL')    
#    defences_after.append(GaussianNoise())
#    defences_names_aft.append('GauN')    
#    defences_after.append(HighConfidence())
#    defences_names_aft.append('HigC')    
#    defences_after.append(ReverseSigmoid())
#    defences_names_aft.append('RevS')
#    defences_after.append(Rounded())
#    defences_names_aft.append('Roud')
    
    '''
    攻击初始化
    '''
    attacks=[]
    attack_names=[]
    eps_L2=[0.1,0.5,1.0,10.0,100.0]
    eps_Linf=[0.005,0.01,0.1,1.0,10.0]
    
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
    读取数据
    '''            
    # 标为原始样本
    fprint_list=[]
    prt_info='\n Clean'
    print(prt_info)
    fprint_list.append(prt_info)
    
            
    images_adv=images.transpose(0,3,1,2)
    predictions = fmodel.predict(images_adv)
    cor_adv = np.sum(np.argmax(predictions,axis=1)==labels)
    prt_info='%s: %.1f'%('van',100*cor_adv/len(labels))
    print(prt_info)
    fprint_list.append(prt_info)
    
    for i in range(len(defences_pre)):
        images_def=images_adv.copy()
        images_in,labels_in = defences_pre[i](images_def.transpose(0,2,3,1),labels.copy())
        predictions = fmodel.predict(images_in.transpose(0,3,1,2))
        cor_adv = np.sum(np.argmax(predictions,axis=1)==labels)
        prt_info='def_pre %s: %.1f'%(defences_names_pre[i],100*cor_adv/len(labels))
        print(prt_info)
        fprint_list.append(prt_info)
        
    for i in range(len(defences_after)):
        predictions     = fmodel.predict(images_adv.copy())
        predictions     = softmax(predictions,axis=1)
        predictions_def = defences_after[i](predictions)
        cor_adv = np.sum(np.argmax(predictions_def,axis=1)==labels)
        prt_info='def_aft %s: %.1f'%(defences_names_aft[i],100*cor_adv/len(labels))
        print(prt_info)
        fprint_list.append(prt_info)

   
    file_log=os.path.join(saved_dir,'result_pre_post_log.txt')
    f=open(file_log,'w')
    f.write('prepost defense result\n')
    for j in range(len(attacks)):
        attack_now=attacks[j]
        prt_info='\n%s'%attack_names[j]
        print(prt_info)
        f=open(file_log,'a')
        f.write(prt_info+'\n')
        f.close()
        fprint_list.append(prt_info)
        
        images_now=images.transpose(0,3,1,2)
        labels_now=labels            
        images_adv  = attack_now.generate(x=images_now)
        predictions = fmodel.predict(images_adv)
        cor_adv = np.sum(np.argmax(predictions,axis=1)==labels_now)
        prt_info='%s: %.1f'%('no_aug',100*cor_adv/len(labels_now))
        print(prt_info)
        f=open(file_log,'a')
        f.write(prt_info+'\n')
        f.close()
        fprint_list.append(prt_info)
        
        for i in range(len(defences_pre)):
            images_def=images_adv.copy()
            images_in,labels_in = defences_pre[i](images_def.transpose(0,2,3,1),labels_now.copy())
            predictions = fmodel.predict(images_in.transpose(0,3,1,2))
            cor_adv = np.sum(np.argmax(predictions,axis=1)==labels_now)
            prt_info='def_pre %s: %.1f'%(defences_names_pre[i],100*cor_adv/len(labels_now))
            print(prt_info)
            f=open(file_log,'a')
            f.write(prt_info+'\n')
            f.close()
            fprint_list.append(prt_info)
            
        for i in range(len(defences_after)):
            predictions     = fmodel.predict(images_adv.copy())
            predictions     = softmax(predictions,axis=1)
            predictions_def = defences_after[i](predictions)
            cor_adv = np.sum(np.argmax(predictions_def,axis=1)==labels_now)
            prt_info='def_aft %s: %.1f'%(defences_names_aft[i],100*cor_adv/len(labels_now))
            print(prt_info)
            f=open(file_log,'a')
            f.write(prt_info+'\n')
            f.close()
            fprint_list.append(prt_info)
        
    
