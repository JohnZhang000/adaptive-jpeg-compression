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
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train

class img_spectrum_labeler:

    # 解释器初始化
    def __init__(self,dataset):
        if 'imagenet'==dataset:
            self.img_size=224
            mean_now=[0.485, 0.456, 0.406]
            std_now=[0.229, 0.224, 0.225]
            self.num_classes=1000
            
        elif 'cifar-10'==dataset:
            self.img_size=32
            mean_now=[0.5] * 3
            std_now=[0.5] * 3
            self.num_classes=10
        else:
            print('ERROR DATASET')
           
        self.trans=transforms.Compose([transforms.Normalize(mean=mean_now, std=std_now)])
        self.s_analyzer=img_spectrum_analyzer(self.img_size).batch_get_spectrum_feature#batch_get_spectrum_energy
        
    def select_attack(self, fmodel, attack_idx, eps_idx):
        attack_names=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP','FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP']
        eps_L2=[0.1,1.0,10.0,100.0]
        eps_Linf=[0.01,0.1,1.0,10.0]
        
        att_method=attack_names[attack_idx]
        if 'L2' in att_method:           
            eps=float(eps_L2[eps_idx%len(eps_L2)])
        else:
            eps=float(eps_Linf[eps_idx%len(eps_Linf)])
            
        if att_method   == 'FGSM_L2_IDP':
            attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=2)
        elif att_method == 'PGD_L2_IDP':
            attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=2,batch_size=512,verbose=False)
        elif att_method == 'CW_L2_IDP':
            attack = CarliniL2Method(classifier=fmodel,batch_size=512,verbose=False)
        elif att_method == 'Deepfool_L2_IDP':
            attack = DeepFool(classifier=fmodel,batch_size=512,verbose=False)
            
        elif att_method == 'FGSM_Linf_IDP':
            attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=np.inf)
        elif att_method == 'PGD_Linf_IDP':
            attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=np.inf,batch_size=512,verbose=False)
        elif att_method == 'CW_Linf_IDP':
            attack = CarliniLInfMethod(classifier=fmodel,eps=eps,batch_size=512,verbose=False)
        
        else:
            raise Exception('Wrong Attack Mode: {} !!!'.format(att_method))
        return attack, eps
        
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
    if len(sys.argv)!=4:
        print('Manual Mode !!!')
        model_type    = 'allconv'
        data          = 'train'
        device        = 2
    else:
        print('Terminal Mode !!!')
        model_type  = sys.argv[1]
        data        = sys.argv[2]
        device      = int(sys.argv[3])
        
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
    dataset='cifar-10'
    dir_model = '../models/cifar_vanilla_'+model_type+'.pth.tar'
    if model_type == 'resnet50_imagenet':
        model = models.resnet50(pretrained=True).eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset = 'imagenet'
    elif model_type == 'vgg16_imagenet':
        model = models.vgg16(pretrained=True).eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset = 'imagenet'
    elif model_type == 'alexnet_imagenet':
        model = models.alexnet(pretrained=True).eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset = 'imagenet'
    elif model_type == 'resnet50':
        model = resnet50().eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)
    elif model_type == 'vgg16':
        model = vgg16_bn().eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)
    elif model_type == 'allconv':
        model = AllConvNet(10).eval()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(dir_model)
        model.load_state_dict(checkpoint["state_dict"],True)  
    else:
        raise Exception('Wrong model name: {} !!!'.format(model_type))
    model.eval()
    
    if 'cifar-10'==dataset:
        mean   = np.array((0.5,0.5,0.5),dtype=np.float32)
        std    = np.array((0.5,0.5,0.5),dtype=np.float32)
        nb_classes = 10
        input_shape=(3,32,32)
        rober_np= np.array([[0,1],[0,2],[0,3],
                        [1,1],[1,2],[1,3],
                        [8,1],[8,2],[8,3],
                        [9,1],[9,2],[9,3],
                        [4,1],[4,2],[4,3],
                        [5,1],[5,2],[5,3],
                        [12,1],[12,2],[12,3],
                        [13,1],[13,2],[13,3],
                       ]) # 被标为rober的设置
    elif 'imagenet'==dataset:
        mean   = np.array((0.485, 0.456, 0.406),dtype=np.float32)
        std    = np.array((0.229, 0.224, 0.225),dtype=np.float32)
        nb_classes = 1000
        input_shape=(3,224,224)
        rober_np= np.array([[0,1],[0,2],[0,3],
                        [1,1],[1,2],[1,3],
                        [8,1],[8,2],[8,3],
                        [9,1],[9,2],[9,3],
                        [4,1],[4,2],[4,3],
                        [5,1],[5,2],[5,3],
                        [12,1],[12,2],[12,3],
                        [13,1],[13,2],[13,3],
                       ]) # 被标为rober的设置
    else:
        raise Exception('Wrong dataset type: {} !!!'.format(dataset))

    fmodel = PyTorchClassifier(model = model,nb_classes=nb_classes,clip_values=(0,1),
                               input_shape=input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(mean, std))  
       
    '''
    读取数据
    '''  
    labeler = img_spectrum_labeler(dataset)
    # fft_transformer = img_transformer(8,0,6)
    
    
    batch           = 10
    batch_num       = int(len(labels)/batch)  
    spectrums_list  = []
    labels_list     = []
    start_time = time.time()
    for i in tqdm(range(batch_num)):
        '''
        攻击与防护
        '''
        images_batch = images[batch*i:batch*(i+1),...].transpose(0,3,1,2)
        labels_batch = labels[batch*i:batch*(i+1),...]
        
        # 标为对抗样本
        attack_name=np.random.randint(1)
        attack_eps=np.random.randint(4)

        attack,eps=labeler.select_attack(fmodel,attack_name, attack_eps)                
        images_adv = attack.generate(x=images_batch,y=labels_batch)
        
        # 根据列表标为有害或无害样本
        flag_rober=0
        for m in range(rober_np.shape[0]):
            if (attack_name==rober_np[m,0]) and (attack_eps==rober_np[m,1]):
                flag_rober=1
                break
        
        spectrums_save,labels_save=labeler.get_energy_label(model, images_adv, labels_batch, flag_rober)
        spectrums_list.append(spectrums_save)
        labels_list.append(labels_save)
                
        
    sub_dir='spectrum_label/'+model_type
    saved_dir_path  = '../saved_tests/img_attack/'+sub_dir
    if not os.path.exists(saved_dir_path):
        os.makedirs(saved_dir_path)
    spectrums_np=np.vstack(spectrums_list)
    labels_np=np.vstack(labels_list)
    
    np.save(os.path.join(saved_dir_path,'spectrums_'+data+'.npy'), spectrums_np)
    np.save(os.path.join(saved_dir_path,'labels_'+data+'.npy'), labels_np)
    end_time=time.time()
    print(("Time %f s")%(end_time-start_time))
    
