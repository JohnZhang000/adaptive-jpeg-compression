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


class img_spectrum_labeler:

    # 解释器初始化
    def __init__(self,dataset):
        if 'imagenet'==dataset:
            self.img_size=g.input_shape_imagenet[2]
            mean_now=g.mean_imagenet
            std_now=g.std_imagenet
            self.num_classes=g.nb_classes_imagenet
            self.spectrum_num=g.spectrum_num_imagenet
            
        elif 'cifar-10'==dataset:
            self.img_size=g.input_shape_cifar[2]
            mean_now=g.mean_cifar
            std_now=g.std_cifar
            self.num_classes=g.nb_classes_cifar
            self.spectrum_num=g.spectrum_num_cifar
        else:
            print('ERROR DATASET')
           
        self.trans=transforms.Compose([transforms.Normalize(mean=mean_now, std=std_now)])
        self.s_analyzer=img_spectrum_analyzer(self.img_size,self.spectrum_num).batch_get_spectrum_feature#batch_get_spectrum_energy
        
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
        
    def get_energy_label(self, model, imgs_in, labels_in, is_adv):
        assert imgs_in.shape[-2]==imgs_in.shape[-1]
        spectrum  = self.s_analyzer(imgs_in)
        labels_out = is_adv * np.ones(labels_in.shape)
        return spectrum,labels_out.reshape((-1,1))

    
    
if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=3:
        print('Manual Mode !!!')
        model_type    = 'allconv'
        data          = 'train'
        # device        = 3
    else:
        print('Terminal Mode !!!')
        model_type  = sys.argv[1]
        data        = sys.argv[2]
        # device      = int(sys.argv[3])
    
    # os.environ['CUDA_VISIBLE_DEVICES']=str(1)
    sub_dir='spectrum_label/'+model_type
    saved_dir_path  = '../saved_tests/img_attack_reg/'+sub_dir
    if not os.path.exists(saved_dir_path):
        os.makedirs(saved_dir_path)
    logging.basicConfig(filename=os.path.join(saved_dir_path,'log.txt'),
                level=logging.FATAL)
    logging.fatal(('\n----------label record-----------'))
    
    '''
    加载模型
    '''
    dir_model = '../models/cifar_vanilla_'+model_type+'.pth.tar'
    model,dataset=g.select_model(model_type, dir_model)
    model.eval()
    
    
    if 'cifar-10'==dataset:
        mean   = g.mean_cifar
        std    = g.std_cifar
        nb_classes = g.nb_classes_cifar
        input_shape=g.input_shape_cifar
        rober_np= g.rober_np_cifar
    elif 'imagenet'==dataset:
        mean   = g.mean_imagenet
        std    = g.std_imagenet
        nb_classes = g.nb_classes_imagenet
        input_shape=g.input_shape_imagenet
        rober_np= g.rober_np_imagenet
    else:
        raise Exception('Wrong dataset type: {} !!!'.format(dataset))

    fmodel = PyTorchClassifier(model = model,nb_classes=nb_classes,clip_values=(0,1),
                               input_shape=input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(mean, std))  
    
    '''
    加载图像
    '''
    if 'cifar-10'==dataset:
        dir_cifar     = g.dir_cifar
        eps_L2=g.eps_L2_label_cifar
        if 'test'==data:
            images,labels = load_CIFAR_batch(os.path.join(dir_cifar,'cifar-10-batches-py/test_batch'))
        elif 'train'==data:
            images,labels = load_CIFAR_train(os.path.join(dir_cifar,'cifar-10-batches-py'))
        else:
            print('Wrong data mode !!!')
    elif 'imagenet'==dataset:
        eps_L2=g.eps_L2_label_imagenet
        with open(g.dir_feature_imagenet) as f:
            features=json.load(f)
        if 'test'==data:
            data_dir=os.path.join(g.dir_imagenet,'val')
        elif 'train'==data:
            data_dir=os.path.join(g.dir_imagenet,'train')
        else:
            print('Wrong data mode !!!')    
        images,labels=load_imagenet_filenames(data_dir,features)
    
    '''
    读取数据
    '''  
    labeler = img_spectrum_labeler(dataset)
    # fft_transformer = img_transformer(8,0,6)
     
    batch           = g.label_batch
    batch_num       = int(len(labels)/batch)  
    spectrums_list  = []
    labels_list     = []
    start_time = time.time()
    for i in tqdm(range(batch_num)):
        '''
        攻击与防护
        '''
        if 'cifar-10'==dataset:
            images_batch = images[batch*i:batch*(i+1),...].transpose(0,3,1,2)
            labels_batch = labels[batch*i:batch*(i+1),...]
            attack_name=0#'FGSM_L2_IDP'
            epss=[0.001,0.1,0.5,1.0,10.0]
            idx=np.random.randint(5)
            attack_eps=epss[idx]#np.random.rand()#np.random.randint(len(eps_L2))
        elif 'imagenet'==dataset:
            images_batch,labels_batch=load_imagenet_batch(i,batch,data_dir,images,labels)
            attack_name=0#'FGSM_L2_IDP'
            attack_eps=np.random.randint(2)
        
        # 标为对抗样本
        attack,eps=g.select_attack(fmodel,g.attack_names[attack_name], attack_eps)                
        images_adv_tmp=attack.generate(x=images_batch,y=labels_batch)
        
        images_ycbcr=g.rgb_to_ycbcr(images_adv_tmp.transpose(0,2,3,1))
        images_dct=g.img2dct(images_ycbcr)
        # #
        # pred=fmodel.predict(images_adv_tmp).argmax(axis=1)
        # succ_attack=(labels_batch!=pred)
        # if sum(succ_attack)==0:
        #     continue
        # else:
        #     images_adv_suc=images_adv_tmp[succ_attack,...]
        #     labels_batch_suc=labels_batch[succ_attack,...]
        
        
        # # 根据列表标为有害或无害样本
        # flag_rober=0
        # for m in range(rober_np.shape[0]):
        #     if (attack_name==rober_np[m,0]) and (attack_eps==rober_np[m,1]):
        #         flag_rober=1
        #         break
        
        # spectrums_save,labels_save=labeler.get_energy_label(model, images_adv_suc, labels_batch_suc, flag_rober)
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
    
    prt_info=("Time of label [%s] [%s] %f s")%(model_type,data,end_time-start_time)
    logging.fatal(prt_info)
    print(prt_info)