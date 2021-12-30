#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:51:35 2021

@author: ubuntu204
"""

import os
import numpy as np
import torch
import torchvision.models as models
from art.attacks.evasion import FastGradientMethod,DeepFool
from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent,BasicIterativeMethod
from art.attacks.evasion import UniversalPerturbation
from foolbox import PyTorchModel
from foolbox.attacks import L2PGD,L2FastGradientAttack
from models.cifar.allconv import AllConvNet
from models.resnet import resnet50
from models.vgg import vgg16_bn
from torchvision import datasets
# from torchvision.datasets import mnist,CIFAR10
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from scipy.fftpack import dct,idct

attack_names=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP','FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP']
eps_L2=[0.1,1.0,10.0]
eps_Linf=[0.01,0.1,1.0,10.0]
      
class dataset_setting():
    def __init__(self,dataset_name='cifar-10'):
        self.dataset_dir=None
        self.mean=None
        self.std=None
        self.nb_classes=None
        self.input_shape=None
        self.pred_batch_size=None
        
        if 'cifar-10'==dataset_name:
            self.dataset_dir='/home/estar/zhangzhuang/Dataset/Cifar'
            self.mean=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.std=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.nb_classes=10
            self.input_shape=(3,32,32)
            self.pred_batch_size=256*4
            
        elif 'imagenet'==dataset_name:
            self.dataset_dir='/home/estar/zhangzhuang/Dataset/Cifar'
            self.mean=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.std=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.nb_classes=10
            self.input_shape=(3,32,32)
            self.pred_batch_size=256
            
        else:
            raise Exception('Wrong dataset')
            
# dir_cifar_img='/media/ubuntu204/F/Dataset/cifar-10'  
dir_cifar='/home/estar/zhangzhuang/Dataset/Cifar'
dir_feature_cifar='../models/cifar-10_class_to_idx.json'
mean_cifar   = np.array((0.5,0.5,0.5),dtype=np.float32)
std_cifar    = np.array((0.5,0.5,0.5),dtype=np.float32)
# mean_cifar   = np.array((0.0,0.0,0.0),dtype=np.float32)
# std_cifar    = np.array((1.0,1.0,1.0),dtype=np.float32)
nb_classes_cifar = 10
input_shape_cifar=(3,32,32)
spectrum_num_cifar=8
# eps_L2_label_cifar=[0.1,1.0,10.0,100.0]
eps_L2_label_cifar=[0.1,0.2,0.3,0.4]
eps_Linf_label_cifar=[0.005,0.01,0.1,1.0]
rober_np_cifar= np.array([[0,1],[0,2],
                [1,1],[1,2],
                [8,1],[8,2],[8,3],
                [9,1],[9,2],[9,3],
                [4,1],[4,2],[4,3],
                [5,1],[5,2],[5,3],
                [12,1],[12,2],[12,3],
                [13,1],[13,2],[13,3],
               ])
levels_all_cifar=8
levels_start_cifar=0
levels_end_cifar=6

shap_batch_cifar=1000
spectrum_batch_cifar=1000

dir_imagenet='../../../../../media/ubuntu204/F/Dataset/ILSVRC2012-10'
dir_feature_imagenet='../models/imagenet_class_to_idx.json'
mean_imagenet   = np.array((0.485, 0.456, 0.406),dtype=np.float32)
std_imagenet    = np.array((0.229, 0.224, 0.225),dtype=np.float32)
nb_classes_imagenet = 1000
input_shape_imagenet=(3,224,224)
spectrum_num_imagenet=8
eps_L2_label_imagenet=[1.0,10.0]
eps_Linf_label_imagenet=[0.1,1.0]
rober_np_imagenet= np.array([[0,1],
                [1,1]
               ]) # 被标为rober的设置
levels_all_imagenet=8
levels_start_imagenet=0
levels_end_imagenet=6

shap_batch_imagenet=500
spectrum_batch_imagenet=500

label_batch=10
pred_batch=100

max_img_in_uap_attack=100

#adaboost params
adb_max_depth=2
adb_epochs=1000

#svm params
svm_gamma=0.1
svm_c=5

#cnn-params
cnn_max_lr     = 3e-4
cnn_epochs     = 300
cnn_batch_size = 256#*16*5
    

def select_attack(fmodel, att_method, eps):
    if att_method   == 'FGSM_L2_IDP':
        attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=2)
        # attack=L2FastGradientAttack()
    elif att_method == 'PGD_L2_IDP':
        # attack = BasicIterativeMethod(estimator=fmodel,eps=eps,eps_step=0.1,batch_size=32,verbose=False)
        attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,eps_step=0.1*eps,batch_size=32,norm=2,verbose=False)
        # attack=L2PGD()
    elif att_method == 'CW_L2_IDP':
        attack = CarliniL2Method(classifier=fmodel,batch_size=32,verbose=False)
    elif att_method == 'Deepfool_L2_IDP':
        attack = DeepFool(classifier=fmodel,batch_size=32,verbose=False)
        
    elif att_method == 'FGSM_Linf_IDP':
        attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=np.inf)
    elif att_method == 'PGD_Linf_IDP':
        attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,eps_step=0.1*eps,norm=np.inf,batch_size=32,verbose=False)
    elif att_method == 'CW_Linf_IDP':
        attack = CarliniLInfMethod(classifier=fmodel,eps=eps,batch_size=32,verbose=False)
    
    elif att_method == 'FGSM_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':2,'verbose':False},max_iter=10,eps=eps,norm=2,batch_size=32,verbose=True)
    elif att_method == 'PGD_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'eps_step':0.1*eps,'norm':2,'verbose':False},max_iter=10,eps=eps,norm=2,batch_size=32,verbose=True)
    elif att_method == 'CW_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='carlini',attacker_params={'eps':eps,'norm':2,'verbose':False},max_iter=10,eps=eps,norm=2,batch_size=32,verbose=True)
    elif att_method == 'Deepfool_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='deepfool',attacker_params={'eps':eps,'norm':2,'verbose':False},max_iter=10,eps=eps,norm=2,batch_size=32,verbose=True)
    
    elif att_method == 'FGSM_Linf_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},max_iter=10,eps=eps,norm=np.inf,batch_size=32,verbose=True)
    elif att_method == 'PGD_Linf_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'eps_step':0.1*eps,'norm':np.inf,'verbose':False},max_iter=10,eps=eps,norm=np.inf,batch_size=32,verbose=True)
    elif att_method == 'CW_Linf_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='carlini_inf',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},max_iter=10,eps=eps,norm=np.inf,batch_size=32,verbose=True)
    
    else:
        raise Exception('Wrong Attack Mode: {} !!!'.format(att_method))
    return attack, eps

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def select_model(model_type,dir_model):
    dataset     ='cifar-10'
    if model_type == 'resnet50_imagenet':
        model = models.resnet50(pretrained=True).eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset ='imagenet'
    elif model_type == 'vgg16_imagenet':
        model = models.vgg16(pretrained=True).eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset ='imagenet'
    elif model_type == 'alexnet_imagenet':
        model = models.alexnet(pretrained=True).eval()
        model = torch.nn.DataParallel(model).cuda()
        dataset ='imagenet'
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
    return model,dataset

def load_dataset(dataset,dataset_dir,dataset_type='train'):
    if 'mnist'==dataset:
        ret_datasets = datasets.mnist.MNIST(root=dataset_dir, train=('train'==dataset_type), transform=ToTensor(), download=True)
    elif 'cifar-10'==dataset:
        # normalize = transforms.Normalize(mean=np.array((0.0,0.0,0.0),dtype=np.float32),
        #                          std=np.array((1.0,1.0,1.0),dtype=np.float32))
        ret_datasets = datasets.CIFAR10(root=dataset_dir, train=('train'==dataset_type), transform=transforms.Compose([ToTensor(),]), download=True)
    elif 'imagenet'==dataset:
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        ret_datasets = datasets.ImageFolder(dataset_dir,
                                            transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                ]))
    else:
        raise Exception('Wrong dataset')
    return ret_datasets

def ycbcr_to_rgb(imgs):
    assert(4==len(imgs.shape))
    assert(imgs.shape[1]==imgs.shape[2])
    
    y=imgs[...,0]
    cb=imgs[...,1]
    cr=imgs[...,2]
    
    delta=0.5
    cb_shift=cb-delta
    cr_shift=cr-delta
    
    r=y+1.403*cr_shift
    g=y-0.714*cr_shift-0.344*cb_shift
    b=y+1.773*cb_shift
    
    imgs_out=np.zeros_like(imgs)
    imgs_out[...,0]=r
    imgs_out[...,1]=g
    imgs_out[...,2]=b
    return imgs_out

def rgb_to_ycbcr(imgs):
    assert(4==len(imgs.shape))
    assert(imgs.shape[1]==imgs.shape[2])
    
    r=imgs[...,0]
    g=imgs[...,1]
    b=imgs[...,2]
    
    delta=0.5
    y=0.299*r+0.587*g+0.114*b
    cb=(b-y)*0.564+delta
    cr=(r-y)*0.713+delta
    
    imgs_out=np.zeros_like(imgs)
    imgs_out[...,0]=y
    imgs_out[...,1]=cb
    imgs_out[...,2]=cr
    return imgs_out

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def img2dct(clean_imgs):
    assert(4==len(clean_imgs.shape))
    assert(clean_imgs.shape[1]==clean_imgs.shape[2])
    n = clean_imgs.shape[0]
    h = clean_imgs.shape[1]
    w = clean_imgs.shape[2]
    c = clean_imgs.shape[3]
    
    block_dct=np.zeros_like(clean_imgs)
    for i in range(n):
        for j in range(c):
            ch_block_cln=clean_imgs[i,:,:,j]                   
            block_cln_tmp = np.log(1+np.abs(dct2(ch_block_cln)))
            block_dct[i,:,:,j]=block_cln_tmp
    return block_dct
