# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:47:03 2021

@author: DELL
"""

import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from my_spectrum_analyzer import img_spectrum_analyzer
import torch
from torchvision import transforms #datasets, models, 
from tqdm import tqdm
import os 
import sys
import torch.nn as nn
import time
from models.cifar.allconv import AllConvNet
from models.resnet import resnet50
from models.vgg import vgg16_bn
from models.cifar.allconv import AllConvNet
import torchvision.models as models
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from art.attacks.evasion import FastGradientMethod,DeepFool
from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
# from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
import json
sys.path.append("..")
# from train_code.my_img_transformer import img_transformer
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.manifold import TSNE
sys.path.append('../common_code')
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train,load_imagenet_batch,load_imagenet_filenames
import general as g
import logging
from torch.utils.data import DataLoader


# class img_spectrum_labeler:

#     # 解释器初始化
#     def __init__(self,dataset):
#         if 'imagenet'==dataset:
#             self.img_size=g.input_shape_imagenet[2]
#             mean_now=g.mean_imagenet
#             std_now=g.std_imagenet
#             self.num_classes=g.nb_classes_imagenet
#             self.spectrum_num=g.spectrum_num_imagenet
            
#         elif 'cifar-10'==dataset:
#             self.img_size=g.input_shape_cifar[2]
#             mean_now=g.mean_cifar
#             std_now=g.std_cifar
#             self.num_classes=g.nb_classes_cifar
#             self.spectrum_num=g.spectrum_num_cifar
#         else:
#             print('ERROR DATASET')
           
#         self.trans=transforms.Compose([transforms.Normalize(mean=mean_now, std=std_now)])
#         self.s_analyzer=img_spectrum_analyzer(self.img_size,self.spectrum_num).batch_get_spectrum_feature#batch_get_spectrum_energy
        
    # def select_attack(self, fmodel, attack_idx, eps_idx):
    #     attack_names=g.attack_names
    #     eps_L2=g.eps_L2
    #     eps_Linf=g.eps_Linf
        
    #     att_method=attack_names[attack_idx]
    #     if 'L2' in att_method:           
    #         eps=float(eps_L2[eps_idx%len(eps_L2)])
    #     else:
    #         eps=float(eps_Linf[eps_idx%len(eps_Linf)])
            
    #     if att_method   == 'FGSM_L2_IDP':
    #         attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=2)
    #     elif att_method == 'PGD_L2_IDP':
    #         attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=2,batch_size=128,verbose=False)
    #     elif att_method == 'CW_L2_IDP':
    #         attack = CarliniL2Method(classifier=fmodel,batch_size=128,verbose=False)
    #     elif att_method == 'Deepfool_L2_IDP':
    #         attack = DeepFool(classifier=fmodel,batch_size=128,verbose=False)
            
    #     elif att_method == 'FGSM_Linf_IDP':
    #         attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=np.inf)
    #     elif att_method == 'PGD_Linf_IDP':
    #         attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=np.inf,batch_size=128,verbose=False)
    #     elif att_method == 'CW_Linf_IDP':
    #         attack = CarliniLInfMethod(classifier=fmodel,eps=eps,batch_size=128,verbose=False)
        
    #     else:
    #         raise Exception('Wrong Attack Mode: {} !!!'.format(att_method))
    #     return attack, eps
        
    # def get_energy_label(self, model, imgs_in, labels_in, is_adv):
    #     assert imgs_in.shape[-2]==imgs_in.shape[-1]
    #     spectrum  = self.s_analyzer(imgs_in)
    #     labels_out = is_adv * np.ones(labels_in.shape)
    #     return spectrum,labels_out.reshape((-1,1))

    
    
if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=3:
        print('Manual Mode !!!')
        model_vanilla_type    = 'resnet50'
        data          = 'test'
        # device        = 3
    else:
        print('Terminal Mode !!!')
        model_vanilla_type  = sys.argv[1]
        data        = sys.argv[2]
        # device      = int(sys.argv[3])
    
    g.setup_seed(0)
    # os.environ['CUDA_VISIBLE_DEVICES']=str(1)
    sub_dir='spectrum_label/'+model_vanilla_type
    saved_dir_path  = '../saved_tests/img_attack/'+model_vanilla_type
    if not os.path.exists(saved_dir_path):
        os.makedirs(saved_dir_path)
    logging.basicConfig(filename=os.path.join(saved_dir_path,'log_label.txt'),
                level=logging.FATAL)
    logging.fatal(('\n----------label record-----------'))
    
    '''
    加载模型
    '''
    dir_model  = '../models/cifar_vanilla_'+model_vanilla_type+'.pth.tar'
    model,dataset_name=g.select_model(model_vanilla_type, dir_model)
    model.eval()
    
        
    '''
    加载图像
    '''
    data_setting=g.dataset_setting(dataset_name)
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,data)
    dataloader = DataLoader(dataset, batch_size=data_setting.label_batch_size, drop_last=False)   
    
    fmodel = PyTorchClassifier(model = model,nb_classes=data_setting.nb_classes,clip_values=(0,1),
                               input_shape=data_setting.input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(data_setting.mean, data_setting.std))

    '''
    读取数据
    '''  
    start_time=time.time()
    spectrums_list=[]
    labels_list=[]
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        
        images=images.numpy()
        labels=labels.numpy()
        
        attack_eps=1*np.random.rand()
        attack,eps=g.select_attack(fmodel,data_setting.hyperopt_attacker_name, attack_eps)                
        images_adv_tmp=attack.generate(x=images,y=labels)
        
        images_ycbcr=g.rgb_to_ycbcr(images_adv_tmp.transpose(0,2,3,1))
        images_dct=g.img2dct(images_ycbcr)
    
        spectrums_list.append(images_dct)
        labels_list.append(eps*np.ones(images_dct.shape[0]))
                
    spectrums_np=np.vstack(spectrums_list)
    labels_np=np.hstack(labels_list)
    
    mean_list=[]
    std_list=[]
    for i in range(spectrums_np.shape[3]):
        mean_list.append(np.expand_dims(spectrums_np[...,i].mean(axis=0),axis=0))
        std_list.append(np.expand_dims(spectrums_np[...,i].std(axis=0),axis=0))
        
    mean_np=np.vstack(mean_list)
    std_np=np.vstack(std_list)
    mean_std=np.vstack((mean_np,std_np)).transpose(1,2,0)
    
    a1=(spectrums_np-mean_std[...,0:3])/mean_std[...,3:6]
    
    np.save(os.path.join(saved_dir_path,'spectrums_'+data+'.npy'), spectrums_np)
    np.save(os.path.join(saved_dir_path,'labels_'+data+'.npy'), labels_np)
    np.save(os.path.join(saved_dir_path,'mean_std_'+data+'.npy'), mean_std)
    end_time=time.time()
    
    prt_info=("Time of label [%s] [%s] %f s")%(model_vanilla_type,data,end_time-start_time)
    logging.fatal(prt_info)
    print(prt_info)