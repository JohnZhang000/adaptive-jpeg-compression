#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:00:21 2021

@author: ubuntu204
"""
from defense_ago import Cal_qtable,Cal_channel_wise_qtable
from tqdm import tqdm

import numpy as np
import torch
import os 
import sys
import torch.nn as nn
from my_data_mining import volcano_mine
import matplotlib.pyplot as plt
import pickle
import cv2
# import torch.nn.functional as F

# from art.attacks.evasion import FastGradientMethod,DeepFool
# from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
# from art.attacks.evasion import ProjectedGradientDescent
# from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
# from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin
# from art.defences.postprocessor import ClassLabels,GaussianNoise,HighConfidence,ReverseSigmoid,Rounded
# from scipy.special import softmax

# from defense import defend_webpf_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
# from defense_ago import defend_FD_ago_warp

# from models.cifar.allconv import AllConvNet
# from third_party.ResNeXt_DenseNet.models.densenet import densenet
# from third_party.ResNeXt_DenseNet.models.resnext import resnext29
# from third_party.WideResNet_pytorch.wideresnet import WideResNet
import json
sys.path.append('../common_code')
# from load_cifar_data import load_CIFAR_batch,load_CIFAR_train
import general as g
# from load_cifar_data import load_CIFAR_batch,load_CIFAR_train,load_imagenet_batch,load_imagenet_filenames

def get_shapleys_batch_adv(attack, model, dataloader, num_samples, device):
    # global id
    # model = model.to(device)
    # model.eval()
    
    dataiter = iter(dataloader)
    
    images = []
    images_adv = []
    num_samples_now = 0
    t = tqdm(range(len(dataloader)))
    for chosen in t:
        t.set_description("Get attacked samples {0:3d}".format(num_samples_now))
        data, label = dataiter.next()
        
        label_now=label.detach().cpu().numpy()#.reshape(-1,1)
        
        # skip not pred true
        data = data.detach().numpy()
        pred = attack.estimator.predict(data)
        pred_class=np.argmax(pred,axis=1)
        correct_pred=(pred_class==label_now)
        if 0==sum(correct_pred):
            continue
        
        if 2==len(correct_pred.shape):
            correct_pred=correct_pred.squeeze(1)
        x=data[correct_pred,...]
        if 3==len(x.shape):
            x=x.unsqueeze(1)
        y=label_now[correct_pred,...]
        img_adv = attack.generate(x)
        img_adv_tc = img_adv
        pred = attack.estimator.predict(img_adv_tc)
        pred_class=np.argmax(pred,axis=1)
        adv_pred=(pred_class!=y)
        if 0==sum(adv_pred):
            continue
        
        if 2==len(adv_pred.shape):
            adv_pred=adv_pred.squeeze(1)
        save_cln=x[adv_pred,...]
        save_adv=img_adv[adv_pred,...]
        images.append(save_cln)
        images_adv.append(save_adv)
        
        num_samples_now=num_samples_now+save_cln.shape[0]
        torch.cuda.empty_cache()
        if num_samples_now>=num_samples:
            break    

    if num_samples_now<num_samples:
        print('\n!!! not enough samples\n')
    
    images_np=None
    images_adv_np=None
    if len(images)>0:
        images_np=np.vstack(images)
    if len(images_adv)>0:
        images_adv_np=np.vstack(images_adv)
    return images_np,images_adv_np

def rgb2ycbcr(imgs):
    imgs_out=np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        img_tmp=imgs[i,...]
        img_tmp=cv2.cvtColor(img_tmp, cv2.COLOR_RGB2YCrCb)
        imgs_out[i,...]=img_tmp
    return imgs_out

if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=4:
        print('Manual Mode !!!')
        thresh0  = 0.3
        thresh1  = 0.8
        thresh2  = 0.8
        # data          = 'test'
        # device        = 0
    else:
        print('Terminal Mode !!!')
        thresh0  = float(sys.argv[1])
        thresh1  = float(sys.argv[2])
        thresh2  = float(sys.argv[3])
        # data        = sys.argv[2]
        # device      = int(sys.argv[3])
    model_vanilla_type    = 'allconv' 
    attacker_name='FGSM_L2_IDP'
    eps=[0.1,0.5,1.0,10.0]
    device=0
    img_num=100
    threshs=[thresh0,thresh1,thresh2]
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    '''
    加载模型
    '''
    dir_model  = '../models/cifar_vanilla_'+model_vanilla_type+'.pth.tar'
    model,dataset=g.select_model(model_vanilla_type, dir_model)
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
        # img_num=g.shap_batch_cifar
    elif 'imagenet'==dataset:
        mean   = g.mean_imagenet
        std    = g.std_imagenet
        nb_classes = g.nb_classes_imagenet
        input_shape=g.input_shape_imagenet
        with open(g.dir_feature_imagenet) as f:
            features=json.load(f)
        fft_level=g.levels_all_imagenet
        dir_img=os.path.join(g.dir_imagenet,'val')
        # img_num=g.shap_batch_imagenet
    else:
        raise Exception('Wrong dataset type: {} !!!'.format(dataset))
    
    fmodel = PyTorchClassifier(model = model,nb_classes=nb_classes,clip_values=(0,1),
                               input_shape=input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(mean, std))

    
    '''
    加载图像
    '''
    g.setup_seed(0)
    datasets=g.load_dataset(dataset,g.dir_cifar)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=10, shuffle=False)
    
    '''
    攻击初始化
    '''
    # threshs=[0.3,0.8,0.8]
    table_dict=dict()
    table_dict[0]=np.ones([8,8,3])
    for eps_now in eps:
        attacker,_=g.select_attack(fmodel,attacker_name,eps_now)
        
        clean_imgs,adv_imgs=get_shapleys_batch_adv(attacker,model,dataloader,img_num,device)
        
        clean_imgs=np.transpose(clean_imgs.copy(),(0,2,3,1))*255
        adv_imgs=np.transpose(adv_imgs.copy(),(0,2,3,1))*255
        clean_imgs_ycc=rgb2ycbcr(clean_imgs)
        adv_imgs_ycc=rgb2ycbcr(adv_imgs)
        
        np.set_printoptions(suppress=True)
        a_qtable,Q,clns,advs=Cal_channel_wise_qtable(clean_imgs_ycc, adv_imgs_ycc,threshs)
        a_qtable=np.round(a_qtable)
        Q=np.round(Q)
        table_dict[eps_now]=a_qtable
    pickle.dump(table_dict, open('table_dict.pkl','wb'))
    
    # print(a_qtable)
    # print(' ')
    # print(Q)
    
    # adv_show=adv_imgs[0,...]
    # cln_show=clean_imgs[0,...]
    
    # # cv2.namedWindow('show')
    # a=adv_show/255
    # a=cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    # cv2.imshow('show', a)
    # cv2.waitKey(0)
    
    # plt.figure()
    # plt.imshow(adv_show)
    # plt.figure()
    # plt.imshow(cln_show)
    
    # plt.figure()
    # plt.imshow((adv_show-cln_show)*50+0.5)
    
    # fc,pc=volcano_mine(np.abs(clns),np.abs(advs))
    # idx_green=(fc>=np.log2(1.2))&(pc>(-np.log10(0.05)))
    # idx_red=(fc<=-np.log2(1.2))&(pc>(-np.log10(0.05)))
    # mine_table=np.zeros([64])
    # mine_table[pc>(-np.log10(0.05))]=1
    # # print(' ')
    # # print(mine_table.reshape([8,8]))
    
    # a=pc.reshape([8,8])
    # a=a/a.max()*80+20
    # a=np.round(a)
    # print(' ')
    # print(a)
    
    
