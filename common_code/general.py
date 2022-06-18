#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:51:35 2021

@author: ubuntu204
"""

import os
import cv2
import numpy as np
import torch
import sys
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
import socket
import PIL
import time
import matplotlib.pyplot as plt

attack_names=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP','FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP']
eps_L2=[0.1,1.0,10.0]
eps_Linf=[0.01,0.1,1.0,10.0]
epsilon=1e-10
      
class dataset_setting():
    def __init__(self,dataset_name='cifar-10'):
        self.dataset_dir=None
        self.mean=None
        self.std=None
        self.nb_classes=None
        self.input_shape=None
        self.pred_batch_size=None
        self.label_batch_size=None
        self.label_eps_range=1
        self.hyperopt_attacker_name='FGSM_L2_IDP'
        self.hyperopt_img_num=1000
        self.hyperopt_img_val_num=None
        self.hyperopt_max_evals=100                                              # modify
        self.hyperopt_thresh_upper=1.0
        self.hyperopt_thresh_lower=0.0
        self.hyperopt_resolution=0.01
        self.early_stoper_patience=10
        
        self.device=socket.gethostname()
        self.cnn_max_lr     = 1e-4
        self.cnn_epochs     = 100
        self.cnn_batch_size = None#*16*5
        self.workers=20
        self.device_num=2
        self.accum_grad_num=1
        self.train_print_epoch=10
        self.eps_L2=[0.1,0.5,1.0]
        
        if 'cifar-10'==dataset_name:
            if 'estar-403'==self.device:
                self.dataset_dir='/home/estar/Datasets/Cifar-10'
                self.workers=20
                self.device_num=2
            elif 'Jet'==self.device:
                self.dataset_dir='/mnt/sdb/zhangzhuang/Datasets/Cifar-10'
                self.workers=32
                self.device_num=3
            elif '1080x4-1'==self.device:
                self.dataset_dir='/home/zhangzhuang/Datasets/cifar-10'
                self.workers=48
                self.device_num=2
            elif 'ubuntu204'==self.device:
                self.dataset_dir='/media/ubuntu204/F/Dataset/cifar-10'
                self.workers=48
                self.device_num=4
            else:
                raise Exception('Wrong device')
            self.mean=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.std=np.array((0.5,0.5,0.5),dtype=np.float32)
            self.nb_classes=10
            self.input_shape=(3,32,32)
            self.pred_batch_size=256
            self.label_batch_size=4
            # self.hyperopt_attacker_name='FGSM_L2_IDP'
            # self.hyperopt_img_num=1000
            # self.hyperopt_img_val_num=0.1
            self.hyperopt_max_evals=1000
            # self.hyperopt_resolution=0.01
            self.cnn_max_lr     = 1e-3
            # self.cnn_epochs     = 300
            self.cnn_batch_size = 256#*16*5
            self.label_eps_range=1
            self.eps_L2=[0.1,0.5,1.0]
            
            
            
        elif 'imagenet'==dataset_name:
            if 'estar-403'==self.device:
                self.dataset_dir='/home/estar/Datasets/ILSVRC2012-100'           # modify
                self.workers=20
                self.device_num=2
                self.pred_batch_size=8
                self.cnn_batch_size =8
            elif 'Jet'==self.device:
                self.dataset_dir='/mnt/sdb/zhangzhuang/Datasets/ILSVRC2012-100'
                self.workers=32
                self.device_num=3
                self.pred_batch_size=8
                self.cnn_batch_size =8
            elif '1080x4-1'==self.device:
                self.dataset_dir='/home/zhangzhuang/Datasets/ILSVRC2012-100'
                self.workers=48
                self.device_num=2
                self.pred_batch_size=8
                self.cnn_batch_size =8
            elif 'ubuntu204'==self.device:
                self.dataset_dir='/media/ubuntu204/F/Dataset/ILSVRC2012-100'
                self.workers=16
                self.device_num=4
                self.pred_batch_size=32
                self.cnn_batch_size =32
            else:
                raise Exception('Wrong device')
            self.mean=np.array((0.485, 0.456, 0.406),dtype=np.float32)
            self.std=np.array((0.229, 0.224, 0.225),dtype=np.float32)
            self.nb_classes=1000
            self.input_shape=(3,224,224)
            # self.pred_batch_size=8
            self.label_batch_size=4
            # self.hyperopt_attacker_name='FGSM_L2_IDP'
            # self.hyperopt_img_num=1000
            self.hyperopt_img_val_num=100#0.1
            self.hyperopt_max_evals=1000
            # self.hyperopt_resolution=0.01
            self.cnn_max_lr     = 1e-7
            # self.cnn_epochs     = 300
            # self.cnn_batch_size = 8#*16*5
            self.label_eps_range=10
            self.accum_grad_num=int(256/self.cnn_batch_size)
            self.eps_L2=[1,5,10]
            
            
        else:
            raise Exception('Wrong dataset')
   

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

def load_dataset(dataset,dataset_dir,dataset_type='train',under_sample=None):
    if 'mnist'==dataset:
        ret_datasets = datasets.mnist.MNIST(root=dataset_dir, train=('train'==dataset_type), transform=ToTensor(), download=True)
    elif 'cifar-10'==dataset:
        # normalize = transforms.Normalize(mean=np.array((0.0,0.0,0.0),dtype=np.float32),
        #                          std=np.array((1.0,1.0,1.0),dtype=np.float32))
        ret_datasets = datasets.CIFAR10(root=dataset_dir, train=('train'==dataset_type), transform=transforms.Compose([ToTensor()]), download=True)
    elif 'imagenet'==dataset:
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        ret_datasets = datasets.ImageFolder(os.path.join(dataset_dir,dataset_type),
                                            transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                ]))
    else:
        raise Exception('Wrong dataset')
    
    if under_sample:
        if under_sample<=1:select_num=int(under_sample*len(ret_datasets))
        else: select_num=under_sample
        left_num=len(ret_datasets)-select_num
        select_datasets,_=torch.utils.data.random_split(ret_datasets, [select_num,left_num])
        ret_datasets=select_datasets
    return ret_datasets

def batch_random_attack(img_t,data_setting,fmodel,mean_std=None):
    imgs=img_t.numpy()
    imgs_dcts=np.zeros_like(imgs)
    eps=np.ones(imgs.shape[0])
    assert(imgs.shape[2]==imgs.shape[3])

    label_batch_size=data_setting.label_batch_size
    label_batch_num=int(np.ceil(imgs.shape[0]/data_setting.label_batch_size))

    # s_time=time.time()
    for i in range(label_batch_num):
        attack_eps=data_setting.label_eps_range*(np.random.rand()+epsilon)
        attack=FastGradientMethod(estimator=fmodel,eps=attack_eps,norm=2)

        start_idx=label_batch_size*i
        end_idx=min(label_batch_size*(i+1),imgs.shape[0])
        imgs_adv=attack.generate(imgs[start_idx:end_idx,...])
        imgs_ycbcr=rgb_to_ycbcr(imgs_adv.transpose(0,2,3,1))
        imgs_dct=img2dct(imgs_ycbcr)
        imgs_dct=imgs_dct.transpose(0,3,1,2)
        imgs_dcts[start_idx:end_idx,...]=imgs_dct
        eps[start_idx:end_idx]=attack_eps
    # e_time=time.time()
    # print('non-pool:%.2f'%(e_time-s_time))

    if not (mean_std is None):
        imgs_dcts=(imgs_dcts-mean_std[0:3,...])/mean_std[3:6,...]
    imgs_dcts=torch.from_numpy(imgs_dcts)
    eps=torch.from_numpy(eps)
    return imgs_dcts,eps

def mp_single_random_attack(imgs_in,data_setting,fmodel):
    attack_eps=data_setting.label_eps_range*(np.random.rand()+epsilon)
    attack=FastGradientMethod(estimator=fmodel,eps=attack_eps,norm=2)

    # s_time_in=time.time()
    imgs_adv=attack.generate(imgs_in)
    # e_time_in=time.time()
    # print('attack:%df'%(e_time_in-s_time_in))

    imgs_ycbcr=rgb_to_ycbcr(imgs_adv.transpose(0,2,3,1))
    imgs_dct=img2dct(imgs_ycbcr)
    imgs_dct=imgs_dct.transpose(0,3,1,2)
    imgs_dct=imgs_adv
    return imgs_dct,attack_eps*np.ones(imgs_in.shape[0])

def mp_batch_random_attack(img_t,data_setting,fmodel,mean_std=None):
    # start pool
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(data_setting.device_num)

    imgs=img_t.numpy()
    imgs_dcts=np.zeros_like(imgs)
    eps=np.ones(imgs.shape[0])
    assert(imgs.shape[2]==imgs.shape[3])

    # s_time=time.time()
    label_batch_size=data_setting.label_batch_size
    label_batch_num=int(np.ceil(imgs.shape[0]/data_setting.label_batch_size))
    pool_list=[]

    for i in range(label_batch_num):
        start_idx=label_batch_size*i
        end_idx=min(label_batch_size*(i+1),imgs.shape[0])
        imgs_in=imgs[start_idx:end_idx,...]
        res=pool.apply_async(mp_single_random_attack,
                            args=(imgs_in,data_setting,fmodel))
        pool_list.append(res)
    
    pool.close()
    pool.join()
    # e_time=time.time()
    # print('pool:%df'%(e_time-s_time))

    imgs_dcts_list=[]
    eps_list=[]
    for i in pool_list:
        res = i.get()
        imgs_dcts_list.append(res[0])
        eps_list.append(res[1])
    imgs_dcts=np.vstack(imgs_dcts_list)
    eps=np.hstack(eps_list)

    if not (mean_std is None):
        imgs_dcts=(imgs_dcts-mean_std[0:3,...])/mean_std[3:6,...]
    imgs_dcts=torch.from_numpy(imgs_dcts)
    eps=torch.from_numpy(eps)
    return imgs_dcts,eps
    
'''  
def load_attacked_dataset(dataset,data_setting,fmodel,dataset_type='train',under_sample=None):
            
    if 'mnist'==dataset:
        ret_datasets = datasets.mnist.MNIST(root=data_setting.dataset_dir, train=('train'==dataset_type), transform=transforms.Compose([random_attack]), download=True)
    elif 'cifar-10'==dataset:
        # normalize = transforms.Normalize(mean=np.array((0.0,0.0,0.0),dtype=np.float32),
        #                          std=np.array((1.0,1.0,1.0),dtype=np.float32))
        ret_datasets = datasets.CIFAR10(root=data_setting.dataset_dir, train=('train'==dataset_type), transform=transforms.Compose([transforms.ToTensor()]), download=True)
    elif 'imagenet'==dataset:
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        ret_datasets = datasets.ImageFolder(os.path.join(data_setting.dataset_dir,dataset_type),
                                            transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                random_attack,
                                                transforms.ToTensor(),
                                                ]))
    else:
        raise Exception('Wrong dataset')
    
    if under_sample:
        select_num=int(under_sample*len(ret_datasets))
        left_num=len(ret_datasets)-select_num
        select_datasets,_=torch.utils.data.random_split(ret_datasets, [select_num,left_num])
        ret_datasets=select_datasets
    return ret_datasets
'''  

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
    # h = clean_imgs.shape[1]
    # w = clean_imgs.shape[2]
    c = clean_imgs.shape[3]
    
    block_dct=np.zeros_like(clean_imgs)
    for i in range(n):
        for j in range(c):
            ch_block_cln=clean_imgs[i,:,:,j]                   
            block_cln_tmp = np.log(1+np.abs(dct2(ch_block_cln)))
            block_dct[i,:,:,j]=block_cln_tmp
    return block_dct

table_y=np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109,103,77],
    [24, 35, 55, 64, 81, 104,113,92],
    [49, 64, 78, 87, 103,121,120,101],
    [72, 92, 95, 98, 112,100,103,99],
    ])

table_c=np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 66, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    ])

def scale_table(table_now,Q=50):
    if Q<=50:
        S=5000/Q
    else:
        S=200-2*Q

    q_table=np.floor((S*table_now+50)/100)
    q_table[q_table==0]=1
    return q_table

def save_images(saved_dir,images,pre_att=None):
    
    assert(images.shape[2]==images.shape[3])
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    images=images.transpose(0,2,3,1)
    for choosed_idx in range(images.shape[0]):
        img_vanilla_tc  = images[choosed_idx,...]
        img_vanilla_np  = np.uint8(np.clip(np.round(img_vanilla_tc*255),0,255))
                
        name=str(choosed_idx)+'.png' 
        img_vanilla_np_res=cv2.resize(img_vanilla_np[...,::-1], (224,224))
        
        saved_name=os.path.join(saved_dir,name)
        if pre_att:
            saved_name=os.path.join(saved_dir,pre_att+name)
        cv2.imwrite(saved_name, img_vanilla_np_res)

def save_images_channel(saved_dir,images,pre_att=None):
    
    assert(images.shape[1]==images.shape[2])
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    for choosed_idx in range(images.shape[0]):
        name=str(choosed_idx) 
        saved_name=os.path.join(saved_dir,name)
        if pre_att:
            saved_name=os.path.join(saved_dir,pre_att+name)
        np.savetxt(saved_name+'.txt',images[choosed_idx,...])

        img_vanilla_tc  = images[choosed_idx,...]
        # img_vanilla_np  = np.uint8(np.clip(np.round(img_vanilla_tc/img_vanilla_tc.max()*255),0,255))
        img_vanilla_np  = np.uint8(np.clip(np.round(img_vanilla_tc/0.05*255),0,255))
        img_vanilla_np_res=cv2.resize(img_vanilla_np, (224,224))
        cv2.imwrite(saved_name+'.png', img_vanilla_np_res)