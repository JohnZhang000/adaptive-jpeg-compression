# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:47:03 2021

@author: DELL
"""

from PIL import Image
import cv2
import gc
from cv2 import transform
import numpy as np
import torch
import os 
import sys
import torch.nn as nn
# import torch.nn.functional as F
sys.path.append('../common_code')
from torch.multiprocessing import Pool, Process, set_start_method


from art.attacks.evasion import FastGradientMethod,DeepFool,AutoAttack,AutoProjectedGradientDescent
from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
# from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin
# from art.defences.postprocessor import ClassLabels,GaussianNoise,HighConfidence,ReverseSigmoid,Rounded
# from scipy.special import softmax

from defense import defend_webpf_wrap,defend_webpf_my_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
from defense import defend_my_webpf
from defense_ago import defend_FD_ago_warp,defend_my_fd_ago
from fd_jpeg import fddnn_defend
from adaptivce_defense import adaptive_defender

from models.cifar.allconv import AllConvNet
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from torch.utils.data import DataLoader
from models.convnext_reg import convnext_xlarge_reg
import torchvision.transforms as transforms
from torch.nn.functional import softmax


import json

# from load_cifar_data import load_CIFAR_batch,load_CIFAR_train
import general as g
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train,load_imagenet_batch,load_imagenet_filenames
import pickle
from tqdm import tqdm
import logging
torch.multiprocessing.set_sharing_strategy('file_system')

class MFIR_defender:

    # 解释器初始化
    def __init__(self,model,mean,std,input_size):
        c,h,w=input_size
        self.model=model
        self.mean=mean
        self.std=std
        if 32==h:
            self.base=2
            self.std1=2
            self.std2=2
        else:
            self.base=20
            self.std1=20
            self.std2=20
        self.transform=transforms.Compose([transforms.RandomResizedCrop((h,w)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean,self.std),])
    
    def pred(self,img):
        img=self.transform(img)
        logits=self.model(img.unsqueeze(0).cuda())
        logits=softmax(logits.detach().cpu(),dim=1)
        return logits.numpy()


    def defend_single(self,img):
        img=np.uint8(img*255)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.bilateralFilter(img,3,self.std1,self.std2)
        img=cv2.bilateralFilter(img,5,self.std1,self.std2)
        logits=[]
        for i in range(10):
            rb=np.random.random()*self.base-self.base/2
            rc=np.random.random()*20-10
            img_tmp=cv2.bilateralFilter(img,self.base-1,1,self.base*i+rb)
            img_tmp=Image.fromarray(cv2.cvtColor(img_tmp,cv2.COLOR_BGR2RGB))
            img_tmp=img_tmp.rotate(rc,expand=1.1)
            logit=self.pred(img_tmp)
            logits.append(logit)
        logits=np.vstack(logits)
        pred=np.argmax(logits.mean(axis=0))
        return pred

    def defend_batch(self,imgs,labels):
        correct=0
        img_num=imgs.shape[0]
        for i in range(img_num):
            pred=self.defend_single(imgs[i])
            if pred==labels[i]: correct+=1
        return correct



def append_attack(attacks,attack,model,epss):
    for i in range(len(epss)):
        attacks.append(attack(estimator=model,eps=epss[i]))   
        
        
def get_acc(fmodel,images,labels):
    with torch.no_grad():
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

# def get_defended_attacked_acc_per_batch(fmodel,attackers,defenders,defender_names,images,labels):
#     cors=np.zeros((len(attackers)+1,len(defenders)+1))
#     for j in range(len(attackers)+1):
#             images_cp=images.copy()
#             labels_cp=labels.copy()
#             images_att=images.copy()
#             eps=0
#             if j>0:
#                 try:
#                     eps=attackers[j-1].eps
#                 except:
#                     eps=0
#                 images_att  = attackers[j-1].generate(x=images_cp)
#             for k in range(len(defenders)+1):
#                     images_def = images_att.copy()
#                     images_att_trs = images_att.transpose(0,2,3,1).copy()
#                     if k>0:
#                         if 'ADAD-flip'==defender_names[k-1]:
#                             images_def,_ = defenders[k-1](images_att_trs,labels_cp,None,0)
#                         elif 'ADAD+eps-flip'==defender_names[k-1]:
#                             images_def,_ = defenders[k-1](images_att_trs,labels_cp,eps*np.ones(images_att.shape[0]),0)
#                         else:
#                             images_def,_ = defenders[k-1](images_att_trs,labels_cp)
#                         images_def=images_def.transpose(0,3,1,2)
#                     images_def_cp = images_def.copy()
#                     cors[j,k] += get_acc(fmodel,images_def_cp,labels)
#                     del images_def,images_def_cp,images_att_trs
#                     gc.collect()
#             del images_cp,images_att,labels_cp
#             gc.collect()
#     return np.expand_dims(cors,axis=0)

# def get_defended_attacked_acc_mp(fmodel,dataloader,attackers,defenders,defender_names):
#     pool_list=[]
#     images_list=[]
#     labels_list=[]
#     for i, (images, labels) in enumerate(tqdm(dataloader)): 
#         res=pool.apply_async(get_defended_attacked_acc_per_batch,
#                             args=(fmodel,attackers,defenders,defender_names,images.numpy(),labels.numpy()))
#         pool_list.append(res)
#     pool.close()
#     pool.join()

#     corss=[]
#     for i in pool_list:
#             cors = i.get()
#             corss.append(cors)
#     cors_np=np.vstack(corss).sum(axis=0)
#     cors=cors_np/len(dataloader.dataset)
#     return cors

def get_defended_attacked_acc(dataloader,attackers,defender):
    cors=np.zeros(len(attackers)+1)
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        images=images.numpy()
        labels=labels.numpy()
        for j in range(len(attackers)+1):
            images_cp=images.copy()
            labels_cp=labels.copy()
            images_att=images.copy()
            eps=0
            if j>0:
                try:
                    eps=attackers[j-1].eps
                except:
                    eps=0
                images_att  = attackers[j-1].generate(x=images_cp)
            images_def = images_att.copy()
            images_att_trs = images_att.transpose(0,2,3,1).copy()
            # images_def_cp = images_def.copy()
            cors[j] += defender(images_att_trs,labels_cp)

            # for k in range(len(defenders)+1):
            #         images_def = images_att.copy()
            #         images_att_trs = images_att.transpose(0,2,3,1).copy()
            #         if k>0:
            #             images_def,_ = defenders[k-1](images_att_trs,labels_cp)
            #             images_def=images_def.transpose(0,3,1,2)
            #         images_def_cp = images_def.copy()
            #         cors[j,k] += get_acc(fmodel,images_def_cp,labels)
            #         del images_def,images_def_cp,images_att_trs
            #         # cors[j,k] += get_acc(fmodel,images_def,labels)
            #         # del images_def,images_att_trs
            #         gc.collect()
            # del images_cp,images_att,labels_cp
            gc.collect()
    cors=cors/len(dataloader.dataset)
    return cors

if __name__=='__main__':    
    '''
    settings
    '''
    # os.environ['CUDA_VISIBLE_DEVICES']='3'
    # 配置解释器参数
    if len(sys.argv)!=2:
        print('Manual Mode !!!')
        model_vanilla_type    = 'vgg16_imagenet'
    else:
        print('Terminal Mode !!!')
        model_vanilla_type    = sys.argv[1]
    
    saved_dir = '../saved_tests/img_attack/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    logger=logging.getLogger(name='r')
    logger.setLevel(logging.FATAL)
    formatter=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    fh=logging.FileHandler(os.path.join(saved_dir,'log_acc_mfir.txt'))
    fh.setLevel(logging.FATAL)
    fh.setFormatter(formatter)
    
    ch=logging.StreamHandler()
    ch.setLevel(logging.FATAL)
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    logger.fatal(('\n----------defense record-----------'))
    
    '''
    加载cifar-10图像
    '''
    g.setup_seed(0)
    if 'imagenet' in model_vanilla_type:
        dataset_name='imagenet'
    else:
        dataset_name='cifar-10'
    data_setting=g.dataset_setting(dataset_name)
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,'val',data_setting.hyperopt_img_val_num)
    dataloader = DataLoader(dataset, batch_size=data_setting.pred_batch_size, drop_last=False, num_workers=data_setting.workers, pin_memory=True)    

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

    defender=MFIR_defender(model,data_setting.mean, data_setting.std,data_setting.input_shape)

    
    
    '''
    攻击初始化
    '''
    attacks=[]
    attack_names=[]
    eps_L2=data_setting.eps_L2                                              # modify
    # eps_L2=[0.1,10.0]
    
    for i in range(len(eps_L2)):
          attacks.append(FastGradientMethod(estimator=fmodel,eps=eps_L2[i],norm=2,eps_step=eps_L2[i]))
          attack_names.append('FGSM_L2_'+str(eps_L2[i]))    
    for i in range(len(eps_L2)):
          attacks.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_L2[i],norm=2,batch_size=data_setting.pred_batch_size,verbose=False))
          attack_names.append('PGD_L2_'+str(eps_L2[i]))    
    attacks.append(DeepFool(classifier=fmodel,batch_size=data_setting.pred_batch_size,verbose=False))
    attack_names.append('DeepFool_L2')    
    attacks.append(CarliniL2Method(classifier=fmodel,batch_size=data_setting.pred_batch_size,verbose=False))
    attack_names.append('CW_L2')
    # for i in range(len(eps_L2)):
    #     attacks.append(AutoAttack(estimator=fmodel,eps=eps_L2[i],eps_step=0.1*eps_L2[i],batch_size=32,norm=2))
    #     attack_names.append('Auto_L2_'+str(eps_L2[i]))    
    # for i in range(len(eps_L2)):
    #     attacks.append(AutoProjectedGradientDescent(estimator=fmodel,eps=eps_L2[i],eps_step=0.1*eps_L2[i],batch_size=32,norm=2))
    #     attack_names.append('AutoPGD_L2_'+str(eps_L2[i]))  
        
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(data_setting.device_num)    

    '''
    计算防御效果
    '''            
    # 标为原始样本
 
    accs=get_defended_attacked_acc(dataloader,attacks,defender.defend_batch)
    np.save(os.path.join(saved_dir,'acc.npy'),accs)
    logger.fatal(attack_names)
    # logger.fatal(defences_names_pre)
    logger.fatal(accs)
    logger.fatal(accs.mean(axis=0))
   