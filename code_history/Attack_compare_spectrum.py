#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:40:34 2021

@author: ubuntu204
"""
import cv2
import torch
import torch.nn as nn
import numpy as np
import os
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
from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
import json
import sys
from tqdm import tqdm
from PIL import Image
sys.path.append("..")
# from train_code.my_img_transformer import img_transformer
# from PathwayGrad.my_pathway_analyzer import my_critical_path
from my_spectrum_analyzer import img_spectrum_analyzer
# from Attack_compare_spectrum_plot import spectrum_analysis
sys.path.append('../common_code')
from load_cifar_data import load_CIFAR_batch

def save_images(saved_dir,vanilla_images,attacked_images,labels,idx,features,attacked_name='attacked'):
    vanilla_dir=os.path.join(saved_dir,'vanilla')
    diff_dir=os.path.join(saved_dir,'diff')
    attacked_dir=os.path.join(saved_dir,attacked_name)
    if not os.path.exists(vanilla_dir):
        os.makedirs(vanilla_dir)
    if not os.path.exists(diff_dir):
        os.makedirs(diff_dir)
    if not os.path.exists(attacked_dir):
        os.makedirs(attacked_dir)
        
    choosed_idx     = 0
    aug_coeff       = 10 #误差增强倍数
    img_vanilla_tc  = vanilla_images[choosed_idx,...].squeeze(0).permute(1,2,0).cpu().numpy()
    img_vanilla_np  = np.uint8(np.clip(np.round(img_vanilla_tc*255),0,255))
    img_attacked_tc = attacked_images[0][choosed_idx,...].squeeze(0).permute(1,2,0).cpu().numpy()
    img_attacked_np = np.uint8(np.clip(np.round(img_attacked_tc*255),0,255))
    img_diff_tc     = (img_attacked_tc - img_vanilla_tc)*aug_coeff
    img_diff_np     = np.uint8(np.clip(np.round((img_diff_tc-img_diff_tc.mean()+0.5)*255),0,255))
    
    label_choosed=list(features.items())[int(labels[choosed_idx].cpu().numpy())][0]
    name=label_choosed+'_'+str(idx+choosed_idx)+'.png' 
    img_vanilla_np_res=cv2.resize(img_vanilla_np, (224,224))
    img_attacked_np_res=cv2.resize(img_attacked_np, (224,224))
    img_diff_np_res=cv2.resize(img_diff_np, (224,224))
    cv2.imwrite(os.path.join(vanilla_dir,name), img_vanilla_np_res)
    cv2.imwrite(os.path.join(diff_dir,name), img_diff_np_res)
    cv2.imwrite(os.path.join(attacked_dir,name), img_attacked_np_res)
    
def pathlist2np(pathlist):
    batch_num=len(pathlist)
    images_num=pathlist[0].shape[0]
    method_num=pathlist[0].shape[1]
    path_num=pathlist[0].shape[2]
    
    paths=np.zeros((batch_num*images_num,method_num,path_num))
    for i in range(batch_num):
        paths[i*images_num:(i+1)*images_num,...]=pathlist[i]
    return paths

def load_imagenet_filenames(dataset_dir,features):
    filename=dataset_dir+'.txt'
    with open(filename, 'r') as f:
        data_list=f.readlines()
        
    label_list=[]
    image_list=[]
    for data in  data_list:
        sysnet,name=data.split('/')
        label_list.append(features[sysnet])
        image_list.append(data.replace('\n',''))
    return image_list,label_list
    
def load_imagenet_batch(batch_idx,batch_size,data_dir,data_list,label_list):
    filenames=data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
    labels=np.array(label_list[batch_idx*batch_size:(batch_idx+1)*batch_size])
    
    images=np.zeros([batch_size,224,224,3])
    for file_idx,file in enumerate(filenames):
        image = Image.open(os.path.join(data_dir,file)).convert('RGB').resize([224,224])
        images[file_idx,...] = np.asarray(image)/255.0
    images=images.transpose(0,3,1,2).astype(np.float32)
    # images=images
    return images,labels
    
                   
'''
settings
'''
# 配置解释器参数
if len(sys.argv)!=5:
    print('Manual Mode !!!')
    model_type    ='resnet50_imagenet'
    att_method    ='FGSM_L2_IDP'
    eps           ='1.0'
    device        = 0
    # print('aaa')
else:
    print('Terminal Mode !!!')
    model_type  = sys.argv[1]
    att_method  = sys.argv[2]
    eps         = sys.argv[3]
    device      = int(sys.argv[4])
    # print('bbb')
    
batch         = 1000
flag_imagenet = 0
# img_per_batch = 2 
# setup_seed(0)
dir_model = '../models/cifar_vanilla_'+model_type+'.pth.tar'

'''
加载模型
'''
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
 
if 'cifar-10'==dataset:
    mean   = np.array((0.5,0.5,0.5),dtype=np.float32)
    std    = np.array((0.5,0.5,0.5),dtype=np.float32)
    nb_classes = 10
    input_shape=(3,32,32)
    with open(os.path.join("../models/cifar-10_class_to_idx.json")) as f:
        features=json.load(f)
    data_dir     = '../../../../../media/ubuntu204/F/Dataset/Dataset_tar/cifar-10-batches-py'
    images,labels = load_CIFAR_batch(os.path.join(data_dir,'test_batch'))
    
elif 'imagenet'==dataset:
    mean   = np.array((0.485, 0.456, 0.406),dtype=np.float32)
    std    = np.array((0.229, 0.224, 0.225),dtype=np.float32)
    nb_classes = 1000
    input_shape=(3,224,224)
    with open(os.path.join("../models/imagenet_class_to_idx.json")) as f:
        features=json.load(f)
    data_dir='../../../../../media/ubuntu204/F/Dataset/ILSVRC2012/val'
    images,labels=load_imagenet_filenames(data_dir,features)
    
else:
    raise Exception('Wrong dataset type: {} !!!'.format(dataset))

fmodel = PyTorchClassifier(model = model,nb_classes=nb_classes,clip_values=(0,1),
                           input_shape=input_shape,loss = nn.CrossEntropyLoss(),
                           preprocessing=(mean, std))

eps=float(eps)
max_img_uni = 50

'''
加载cifar-10图像
'''
# os.environ['CUDA_VISIBLE_DEVICES']=str(device)


'''
攻击初始化
'''   
if att_method   == 'FGSM_L2_IDP':
    attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=2)
elif att_method == 'PGD_L2_IDP':
    attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=2,batch_size=128,verbose=False)
elif att_method == 'CW_L2_IDP':
    attack = CarliniL2Method(classifier=fmodel,batch_size=128,verbose=False)
elif att_method == 'Deepfool_L2_IDP':
    attack = DeepFool(classifier=fmodel,batch_size=128,verbose=False)
    
elif att_method == 'FGSM_Linf_IDP':
    attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=np.inf)
elif att_method == 'PGD_Linf_IDP':
    attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=np.inf,batch_size=128,verbose=False)
elif att_method == 'CW_Linf_IDP':
    attack = CarliniLInfMethod(classifier=fmodel,eps=eps,batch_size=128,verbose=False)

elif att_method == 'FGSM_L2_UAP':
    attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=128,verbose=False)
    batch = max_img_uni
elif att_method == 'PGD_L2_UAP':
    attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=128,verbose=False)
    batch = max_img_uni
elif att_method == 'CW_L2_UAP':
    attack = UniversalPerturbation(classifier=fmodel,attacker='carlini',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=128,verbose=True)
    batch = max_img_uni
elif att_method == 'Deepfool_L2_UAP':
    attack = UniversalPerturbation(classifier=fmodel,attacker='deepfool',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=128,verbose=False)
    batch = max_img_uni

elif att_method == 'FGSM_Linf_UAP':
    attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=128,verbose=False)
    batch = max_img_uni
elif att_method == 'PGD_Linf_UAP':
    attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=128,verbose=False)
    batch = max_img_uni
elif att_method == 'CW_Linf_UAP':
    attack = UniversalPerturbation(classifier=fmodel,attacker='carlini_inf',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=128,verbose=False)
    batch = max_img_uni

else:
    raise Exception('Wrong Attack Mode: {} !!!'.format(att_method))
   
'''
读取数据
'''  
# fft_transformer = img_transformer(8,0,6)
saved_dir       = '../saved_tests/img_attack/'+model_type+'_'+att_method+'_'+str(eps)

success_num     = 0
clean_accs      = 0
masked_cln_accs = 0
masked_adv_accs = 0
jpg_cln_accs    = 0
jpg_adv_accs    = 0
pca_cln_accs    = 0
pca_adv_accs    = 0
pcab_cln_accs   = 0
pcab_adv_accs   = 0
model_sparsity_threshold = None
batch_num       = int(len(labels)/batch)
# pather          = my_critical_path(model, 80, 'cifar-10')
s_analyzer      = img_spectrum_analyzer(input_shape[1])
map_diff_cln    = []
map_diff_adv    = []
map_diff_msk    = []
clean_path_list = []
adv_path_list   = []
msk_clean_path_list = []
msk_adv_path_list   = []
Es_cln_list     = []
Es_adv_list     = []
Es_mcln_list    = []
Es_madv_list    = []

Es_diff_list    = []
Es_diff_list_mcln    = []
Es_diff_list_madv    = []
print(batch)
for i in tqdm(range(batch_num)):
    if i>0:
        continue
    '''
    攻击与防护
    '''
    if 'cifar-10'==dataset:
        images_batch = images[batch*i:batch*(i+1),...].transpose(0,3,1,2)
        labels_batch = labels[batch*i:batch*(i+1),...]
    elif 'imagenet'==dataset:
        images_batch,labels_batch=load_imagenet_batch(i,batch,data_dir,images,labels)
    
    # 原始准确率
    predictions = fmodel.predict(images_batch)
    clean_accs += np.sum(np.argmax(predictions,axis=1)==labels_batch)
       
    # 攻击
    img_adv    = attack.generate(x=images_batch,y=labels_batch)
    predictions = fmodel.predict(img_adv)
    success_num += np.sum(np.argmax(predictions,axis=1)!=labels_batch)

    # # 低频滤波 clean
    # masked_clns  = fft_transformer.img_transform_tc(images_batch)
    # predictions = fmodel.predict(masked_clns)
    # masked_cln_accs += np.sum(np.argmax(predictions,axis=1)==labels_batch)
     
    # # 低频滤波 adv
    # masked_advs  = fft_transformer.img_transform_tc(img_adv)
    # predictions = fmodel.predict(masked_advs)
    # masked_adv_accs += np.sum(np.argmax(predictions,axis=1)==labels_batch)
    
    '''
    频谱分析
    '''
    E,_ = s_analyzer.batch_get_spectrum_energy(images_batch)
    Es_cln_list.append(E)
    
    E,_ = s_analyzer.batch_get_spectrum_energy(img_adv)
    Es_adv_list.append(E)
    
    # E,_ = s_analyzer.batch_get_spectrum_energy(masked_clns)
    # Es_mcln_list.append(E)
    
    # E,_ = s_analyzer.batch_get_spectrum_energy(masked_advs)
    # Es_madv_list.append(E)
    
        
    E,_ = s_analyzer.batch_get_spectrum_energy((img_adv-images_batch))
    Es_diff_list.append(E)
    
    # E,_ = s_analyzer.batch_get_spectrum_energy((masked_clns-images_batch))
    # Es_diff_list_mcln.append(E)
    
    # E,_ = s_analyzer.batch_get_spectrum_energy((masked_advs-images_batch))
    # Es_diff_list_madv.append(E)
    
    torch.cuda.empty_cache()
    
sub_dir='spectrum'
saved_dir_path  = '../saved_tests/img_attack/'+model_type+'_'+att_method+'_'+str(eps)+'/'+sub_dir
if not os.path.exists(saved_dir_path):
    os.makedirs(saved_dir_path)
Es_cln_np=np.vstack(Es_cln_list)
Es_adv_np=np.vstack(Es_adv_list)
# Es_mcln_np=np.vstack(Es_mcln_list)
# Es_madv_np=np.vstack(Es_madv_list)
Es_diff_np=np.vstack(Es_diff_list)
# Es_diff_np_mcln=np.vstack(Es_diff_list_mcln)
# Es_diff_np_madv=np.vstack(Es_diff_list_madv)

np.save(os.path.join(saved_dir_path,'clean_spectrum.npy'), Es_cln_np)
np.save(os.path.join(saved_dir_path,'adv_spectrum.npy'), Es_adv_np)
# np.save(os.path.join(saved_dir_path,'mclean_spectrum.npy'), Es_mcln_np)
# np.save(os.path.join(saved_dir_path,'madv_spectrum.npy'), Es_madv_np)

np.save(os.path.join(saved_dir_path,'diff_spectrum.npy'), Es_diff_np)
# np.save(os.path.join(saved_dir_path,'mcln_diff_spectrum.npy'), Es_diff_np_mcln)
# np.save(os.path.join(saved_dir_path,'madv_diff_spectrum.npy'), Es_diff_np_madv)

