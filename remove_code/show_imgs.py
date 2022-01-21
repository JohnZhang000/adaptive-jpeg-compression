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
from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin
from art.defences.postprocessor import ClassLabels,GaussianNoise,HighConfidence,ReverseSigmoid,Rounded
from defense import defend_webpf_wrap,defend_webpf_my_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
from defense import defend_my_webpf
from defense_ago import defend_FD_ago_warp,defend_my_fd_ago
from fd_jpeg import fddnn_defend
from adaptivce_defense import adaptive_defender
import pickle
 
def get_spectrum(imgs):
    images_ycbcr=g.rgb_to_ycbcr(imgs.transpose(0,2,3,1))
    images_dct=g.img2dct(images_ycbcr)
    return images_dct

if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=3:
        print('Manual Mode !!!')
        model_vanilla_type    = 'allconv'
        data          = 'val'
        # device        = 3
    else:
        print('Terminal Mode !!!')
        model_vanilla_type  = sys.argv[1]
        data        = sys.argv[2]
        # device      = int(sys.argv[3])
    
    data_num=1000
    g.setup_seed(0)
    # os.environ['CUDA_VISIBLE_DEVICES']=str(1)
    sub_dir='spectrum_label/'+model_vanilla_type
    saved_dir_path  = '../saved_tests/img_attack/'+model_vanilla_type+'/imgs'
    if not os.path.exists(saved_dir_path):
        os.makedirs(saved_dir_path)
    logging.basicConfig(filename=os.path.join(saved_dir_path,'log_show.txt'),
                level=logging.FATAL)
    logging.fatal(('\n----------label show-----------'))
    
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
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,data,data_setting.hyperopt_img_val_num)
    dataloader = DataLoader(dataset, batch_size=data_setting.pred_batch_size, drop_last=False, num_workers=data_setting.workers, pin_memory=True)   
    
    fmodel = PyTorchClassifier(model = model,nb_classes=data_setting.nb_classes,clip_values=(0,1),
                               input_shape=data_setting.input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(data_setting.mean, data_setting.std))

    '''
    攻击初始化
    '''
    attacks=[]
    attack_names=[]
    eps_L2=data_setting.eps_L2
    
    for i in range(len(eps_L2)):
          attacks.append(FastGradientMethod(estimator=fmodel,eps=eps_L2[i],norm=2,eps_step=eps_L2[i]))
          attack_names.append('FGSM_L2_'+str(eps_L2[i]))    
    # for i in range(len(eps_L2)):
    #       attacks.append(ProjectedGradientDescent(estimator=fmodel,eps=eps_L2[i],norm=2,batch_size=data_setting.pred_batch_size,verbose=False))
    #       attack_names.append('PGD_L2_'+str(eps_L2[i]))    
    # attacks.append(DeepFool(classifier=fmodel,batch_size=data_setting.pred_batch_size,verbose=False))
    # attack_names.append('DeepFool_L2')    
    # attacks.append(CarliniL2Method(classifier=fmodel,batch_size=data_setting.pred_batch_size,verbose=False))
    # attack_names.append('CW_L2')

    '''
    防御初始化
    '''
    defences_pre=[]
    defences_names_pre=[]
    defences_pre.append(GaussianAugmentation(sigma=0.01,augmentation=False))
    defences_names_pre.append('GauA')
    defences_pre.append(SpatialSmoothing())
    defences_names_pre.append('BDR')
    defences_pre.append(defend_rdg_wrap)
    defences_names_pre.append('RDG')
    defences_pre.append(defend_webpf_wrap)
    defences_names_pre.append('WEBPF')
    defences_pre.append(JpegCompression(clip_values=(0,1),quality=25,channels_first=False))
    defences_names_pre.append('JPEG')
    defences_pre.append(defend_shield_wrap)
    defences_names_pre.append('SHIELD')
    defences_pre.append(defend_fd_wrap)
    defences_names_pre.append('EPA')
    defences_pre.append(fddnn_defend)
    defences_names_pre.append('FD')
    # # defences_pre.append(defend_FD_ago_warp)
    # # defences_names_pre.append('FD_ago')
    
    table_pkl=os.path.join('../saved_tests/img_attack/'+model_vanilla_type,'table_dict.pkl')
    gc_model_dir=os.path.join('../saved_tests/img_attack/'+model_vanilla_type,'model_best.pth.tar')
    model_mean_std=os.path.join('../saved_tests/img_attack/'+model_vanilla_type,'mean_std_train.npy')
    adaptive_defender=adaptive_defender(table_pkl,gc_model_dir,data_setting.nb_classes,data_setting.input_shape[-1],data_setting.pred_batch_size,model_mean_std)
    
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD')
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD-flip')
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD+eps-flip')
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD+eps+flip')

    '''
    读取数据
    '''  
    start_time=time.time()
    
    spectrums=np.zeros((len(attacks)+1,data_num,data_setting.input_shape[1],data_setting.input_shape[2],data_setting.input_shape[0]))
    images_clns=np.zeros((data_num,data_setting.input_shape[1],data_setting.input_shape[2],data_setting.input_shape[0]))
    images_advs=np.zeros((len(attacks),data_num,data_setting.input_shape[1],data_setting.input_shape[2],data_setting.input_shape[0]))
    images_defs=np.zeros((len(attacks),len(defences_pre),data_num,data_setting.input_shape[1],data_setting.input_shape[2],data_setting.input_shape[0]))
    images_defs_spectrum=np.zeros((len(attacks),len(defences_pre),data_num,data_setting.input_shape[1],data_setting.input_shape[2],data_setting.input_shape[0]))

    start_idx=0
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        
        images=images.numpy()
        labels=labels.numpy()

        if start_idx+len(labels) > data_num:
            images=images[0:data_num-start_idx]
            labels=labels[0:data_num-start_idx]

        images_dct=get_spectrum(images.copy())

        spectrums[0,start_idx:start_idx+len(labels),...]=images_dct
        images_clns[start_idx:start_idx+len(labels),...]=images.transpose(0,2,3,1)

        for j in range(len(attacks)):               
            images_adv_tmp=attacks[j].generate(x=images.copy(),y=labels)
        
            images_dct=get_spectrum(images_adv_tmp.copy())
        
            spectrums[j+1,start_idx:start_idx+len(labels),...]=images_dct
            images_advs[j,start_idx:start_idx+len(labels),...]=images_adv_tmp.transpose(0,2,3,1)

            images_adv_tmp=images_adv_tmp.transpose(0,2,3,1)
            for k in range(len(defences_pre)):
                if 'ADAD-flip'==defences_names_pre[k]:
                    images_def,_ = defences_pre[k](images_adv_tmp.copy(),labels,None,0)
                elif 'ADAD+eps-flip'==defences_names_pre[k]:
                    images_def,_ = defences_pre[k](images_adv_tmp.copy(),labels,attacks[j].eps*np.ones(images_adv_tmp.shape[0]),0)
                elif 'ADAD+eps+flip'==defences_names_pre[k]:
                    images_def,_ = defences_pre[k](images_adv_tmp.copy(),labels,attacks[j].eps*np.ones(images_adv_tmp.shape[0]),1)
                else:
                    images_def,_ = defences_pre[k](images_adv_tmp.copy(),labels)
                images_def=images_def.transpose(0,3,1,2)
                images_defs[j,k,start_idx:start_idx+len(labels),...]=images_def.transpose(0,2,3,1)

                images_dct=get_spectrum(images_def.copy())
                images_defs_spectrum[j,k,start_idx:start_idx+len(labels),...]=images_dct-spectrums[j+1,start_idx:start_idx+len(labels),...]

        start_idx=start_idx+len(labels)
        if start_idx>=data_num:
            break
    print('done')

    # '''
    # 输出图片
    # '''
    # images_clns=images_clns.transpose(0,3,1,2) 
    # images_advs=images_advs.transpose(0,1,4,2,3)
    # images_defs=images_defs.transpose(0,1,2,5,3,4)
    # g.save_images(os.path.join(saved_dir_path,'images/cleans'),images_clns) 
    # for i in range(images_advs.shape[0]):
    #     g.save_images(os.path.join(saved_dir_path,'images/advs/'+attack_names[i]+'/attacked'),images_advs[i,...])
    #     for j in range(images_defs.shape[1]):
    #         g.save_images(os.path.join(saved_dir_path,'images/advs/'+attack_names[i]+'/'+defences_names_pre[j]),images_defs[i,j,...])
            
    '''
    输出扰动
    '''
    # images_diffs=images_advs-images_clns 
    # images_diffs=images_diffs.mean(axis=1).transpose(0,3,1,2)   
    # for i in range(images_diffs.shape[0]):
    #     g.save_images_channel(os.path.join(saved_dir_path,'perturbations'),images_diffs[i,...],str(i))

    '''
    输出频谱
    '''
    # spectrums_show=spectrums.mean(axis=1) 
    # spectrums_show=spectrums_show.transpose(0,3,1,2)   
    # for i in range(spectrums_show.shape[0]):
    #     g.save_images_channel(os.path.join(saved_dir_path,'spectrums'),spectrums_show[i,...],str(i))

    # for i in range(spectrums_show.shape[0]-1):
    #     g.save_images_channel(os.path.join(saved_dir_path,'spectrums'),spectrums_show[i+1,...]-spectrums_show[0,...],'diff_'+str(i+1))  

    '''
    输出防御后的频谱
    '''
    spectrums_show=images_defs_spectrum.mean(axis=2) 
    spectrums_show=np.abs(spectrums_show).transpose(0,1,4,2,3)   
    for i in range(spectrums_show.shape[0]):
        for j in range(spectrums_show.shape[1]):
            pre_fix=attack_names[i]+'_'+defences_names_pre[j]+'_'
            pre_fix=pre_fix.replace('FGSM_L2_','')
            g.save_images_channel(os.path.join(saved_dir_path,'spectrums/defenders'),spectrums_show[i,j,...],pre_fix)
    
    # '''
    # 输出表格
    # '''
    # table_saved_dir=os.path.join(saved_dir_path,'tables')
    # if not os.path.exists(table_saved_dir):
    #     os.makedirs(table_saved_dir)
    # table_dict=pickle.load(open(table_pkl,'rb'))
    # for key, value in table_dict.items():
    #     for i in range(value.shape[2]):
    #         np.savetxt(os.path.join(table_saved_dir,str(key)+'_'+str(i)+'.txt'),value[...,i])

