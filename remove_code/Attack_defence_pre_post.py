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
from defense_ago import defend_FD_ago_warp

from models.cifar.allconv import AllConvNet
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
import json
sys.path.append('../common_code')
# from load_cifar_data import load_CIFAR_batch,load_CIFAR_train
import general as g
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train,load_imagenet_batch,load_imagenet_filenames


def append_attack(attacks,attack,model,epss):
    for i in range(len(epss)):
        attacks.append(attack(estimator=model,eps=epss[i]))   

if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=2:
        print('Manual Mode !!!')
        model_vanilla_type    = 'allconv'
        # data          = 'test'
        # device        = 0
    else:
        print('Terminal Mode !!!')
        model_vanilla_type  = sys.argv[1]
        # data        = sys.argv[2]
        # device      = int(sys.argv[3])
        
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    '''
    加载cifar-10图像
    '''
    g.setup_seed(0)
    # dir_cifar     = g.dir_cifar
    # images,labels = load_CIFAR_batch(os.path.join(dir_cifar,'test_batch'))
    if 'imagenet' in model_vanilla_type:
        dataset='imagenet'
    else:
        dataset='cifar-10'
    if 'cifar-10'==dataset:
        dir_cifar     = g.dir_cifar
        images,labels = load_CIFAR_batch(os.path.join(dir_cifar,'cifar-10-batches-py/test_batch'))
        images=images.transpose(0,3,1,2)
    elif 'imagenet'==dataset:
        with open(g.dir_feature_imagenet) as f:
            features=json.load(f)
        data_dir=os.path.join(g.dir_imagenet,'val')
        images_names,labels=load_imagenet_filenames(data_dir,features)
        
        batch_size=g.pred_batch
        batch_num       = int(len(labels)/batch_size) 
        images_list=[]
        label_list=[]
        for i in range(batch_num):
            start_idx=batch_size*i
            end_idx=min(batch_size*(i+1),len(labels))
            images_tmp,labels_tmp=load_imagenet_batch(i,batch_size,data_dir,images_names,labels)
            images_list.append(images_tmp)
            label_list.append(labels_tmp)
        images=np.vstack(images_list)
        labels=np.hstack(label_list)

    '''
    加载模型
    '''
    dir_model  = '../models/cifar_vanilla_'+model_vanilla_type+'.pth.tar'
    model,_=g.select_model(model_vanilla_type, dir_model)
    model.eval()
    
    if 'cifar-10'==dataset:
        mean   = g.mean_cifar
        std    = g.std_cifar
        nb_classes = g.nb_classes_cifar
        input_shape=g.input_shape_cifar
        with open(g.dir_feature_cifar) as f:
            features=json.load(f)
        fft_level=g.levels_all_cifar
        dir_img=os.path.join(g.dir_cifar,'val')
        img_num=g.shap_batch_cifar
    elif 'imagenet'==dataset:
        mean   = g.mean_imagenet
        std    = g.std_imagenet
        nb_classes = g.nb_classes_imagenet
        input_shape=g.input_shape_imagenet
        with open(g.dir_feature_imagenet) as f:
            features=json.load(f)
        fft_level=g.levels_all_imagenet
        dir_img=os.path.join(g.dir_imagenet,'val')
        img_num=g.shap_batch_imagenet
    else:
        raise Exception('Wrong dataset type: {} !!!'.format(dataset))
    
    fmodel = PyTorchClassifier(model = model,nb_classes=nb_classes,clip_values=(0,1),
                               input_shape=input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(mean, std))

    
    pred_cln = fmodel.predict(images)
    torch.cuda.empty_cache()
   
    '''
    防御初始化
    '''

    defences_pre=[]
    defences_names_pre=[]
    # defences_pre.append(JpegCompression(clip_values=(0,1),quality=25,channels_first=False))
    # defences_names_pre.append('JPEG')
    #defences_pre.append(FeatureSqueezing(clip_values=(0,1),bit_depth=32))
    #defences_names_pre.append('FeaS')
    # defences_pre.append(GaussianAugmentation(sigma=0.01,augmentation=False))
    # defences_names_pre.append('GauA')
    # defences_pre.append(LabelSmoothing(apply_predict=True))
    # defences_names_pre.append('LabS')
    # defences_pre.append(Resample(sr_original=16000,sr_new=8000,channels_first=False))
    # defences_names_pre.append('Resa')
    # defences_pre.append(SpatialSmoothing())
    # defences_names_pre.append('SpaS')
    #defences_pre.append(ThermometerEncoding(clip_values=(0,1),num_space=1))
    #defences_names_pre.append('TheE')
    #defences_pre.append(TotalVarMin())
    #defences_names_pre.append('ToVM')
    # defences_pre.append(defend_webpf_my_wrap)
    # defences_names_pre.append('webpf_my')
    defences_pre.append(defend_webpf_wrap)
    defences_names_pre.append('webpf')
    # defences_pre.append(defend_rdg_wrap)
    # defences_names_pre.append('rdg')
    # defences_pre.append(defend_fd_wrap)
    # defences_names_pre.append('fd')
    # defences_pre.append(defend_bdr_wrap)
    # defences_names_pre.append('bdr')
    # defences_pre.append(defend_shield_wrap)
    # defences_names_pre.append('shield')
    defences_pre.append(defend_FD_ago_warp)
    defences_names_pre.append('FD_ago')
    
    # model_pkl='../saved_tests/img_attack_reg/spectrum_label/allconv/allconv_other.pkl'
    # webpf_new=defend_my_webpf(model_pkl,32,8)
    # defences_pre.append(webpf_new.defend)
    # defences_names_pre.append('webpf_my_new')
    
    # if Q<50:
    #     S=5000/Q
    # else:
    #     S=200-2*Q
    # end
        
    # Ts=floor((S*Tb+50)/100)
    # Ts(Ts==0)=1
    
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
    
    
    # for i in range(len(eps_Linf)):
    #     attacks.append(FastGradientMethod(estimator=fmodel,eps=eps_Linf[i],norm=np.inf,eps_step=eps_Linf[i]))
    #     attack_names.append('FGSM_Linf_'+str(eps_Linf[i]))    
    # for i in range(len(eps_Linf)):
    #     attacks.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_Linf[i],norm=np.inf,batch_size=512,verbose=False))
    #     attack_names.append('PGD_Linf_'+str(eps_Linf[i]))     
    # for i in range(len(eps_Linf)):
    #     attacks.append(CarliniLInfMethod(classifier=fmodel,batch_size=512,eps=eps_Linf[i],verbose=False))
    #     attack_names.append('CW_Linf_'+str(eps_Linf[i]))   
        

    '''
    读取数据
    '''            
    # 标为原始样本
    fprint_list=[]
    prt_info='\n Clean'
    print(prt_info)
    fprint_list.append(prt_info)
    
            
    images_adv=images
    predictions = fmodel.predict(images_adv)
    predictions = np.argmax(predictions,axis=1)
    cor_adv = np.sum(predictions==labels)
    prt_info='%s: %.1f'%('van',100*cor_adv/len(labels))
    print(prt_info)
    fprint_list.append(prt_info)
    
    for i in range(len(defences_pre)):
        images_def=images_adv.copy()
        if 'webpf_my'==defences_names_pre[i]:
            images_in,labels_in = defences_pre[i](images_def.transpose(0,2,3,1),0*np.ones(images_def.shape[0]),labels.copy())
        else:
            images_in,labels_in = defences_pre[i](images_def.transpose(0,2,3,1),labels.copy())
        predictions = fmodel.predict(images_in.transpose(0,3,1,2))
        predictions = np.argmax(predictions,axis=1)
        cor_adv = np.sum(predictions==labels)
        prt_info='def_pre %s: %.1f'%(defences_names_pre[i],100*cor_adv/len(labels))
        print(prt_info)
        fprint_list.append(prt_info)
        torch.cuda.empty_cache()
        del images_def

   
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
        
        images_now=images
        labels_now=labels    
        images_adv_list=[]  
        predictions_list=[]  
        batch_num       = int(len(labels_now)/g.label_batch) 
        images_adv_list=[]
        for i_attack in range(batch_num):
            start_idx=g.label_batch*i_attack
            end_idx=min(g.label_batch*(i_attack+1),len(labels_now))
            images_adv_tmp  = attack_now.generate(x=images_now[start_idx:end_idx,...])
            predictions_tmp = fmodel.predict(images_adv_tmp)
            predictions_tmp = np.argmax(predictions_tmp,axis=1)
            images_adv_list.append(images_adv_tmp)
            predictions_list.append(predictions_tmp)
            torch.cuda.empty_cache()
        
        images_adv  = np.vstack(images_adv_list)
        predictions = np.hstack(predictions_list)
        cor_adv = np.sum(predictions==labels_now)
        prt_info='%s: %.1f'%('no_aug',100*cor_adv/len(labels_now))
        print(prt_info)
        f=open(file_log,'a')
        f.write(prt_info+'\n')
        f.close()
        fprint_list.append(prt_info)
        
        for i in range(len(defences_pre)):
            images_def=images_adv.copy()
            if 'webpf_my'==defences_names_pre[i]:
                if 'Deepfool' in attack_names[j] or 'CW' in attack_names[j]:
                    eps_pred=0.1
                else:
                    eps_pred=attack_now.eps
                images_in,labels_in = defences_pre[i](images_def.transpose(0,2,3,1),eps_pred*np.ones(images_def.shape[0]),labels.copy())
            else:
                images_in,labels_in = defences_pre[i](images_def.transpose(0,2,3,1),labels.copy())
            predictions = fmodel.predict(images_in.transpose(0,3,1,2))
            predictions = np.argmax(predictions,axis=1)
            cor_adv = np.sum(predictions==labels_now)
            prt_info='def_pre %s: %.1f'%(defences_names_pre[i],100*cor_adv/len(labels_now))
            print(prt_info)
            f=open(file_log,'a')
            f.write(prt_info+'\n')
            f.close()
            fprint_list.append(prt_info)
            torch.cuda.empty_cache()
            del images_def

        
    
