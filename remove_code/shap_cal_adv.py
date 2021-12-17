# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:24:49 2020

@author: DELL
"""
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' #只显示警告和报错
import torch
import torchvision.models as models
# import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
# from skimage.segmentation import slic
from fft_loc_shapley_explainer_pytorch2 import wave_shapley_explainer
# import time

import matplotlib.pyplot as plt
# import csv
import random
# import torchvision

import pandas as pd
import datetime
from tqdm import tqdm
import sys
import cv2
# import cupy as cp
import json

from models.cifar.allconv import AllConvNet
from models.resnet import resnet50
from models.vgg import vgg16_bn
# from third_party.ResNeXt_DenseNet.models.densenet import densenet
# from third_party.ResNeXt_DenseNet.models.resnext import resnext29
# from third_party.WideResNet_pytorch.wideresnet import WideResNet

# from foolbox import PyTorchModel, accuracy, samples
# from foolbox.attacks import FGM,L2PGD,L2CarliniWagnerAttack,L2DeepFoolAttack,L2BrendelBethgeAttack,L2ClippingAwareAdditiveUniformNoiseAttack,L2ClippingAwareAdditiveGaussianNoiseAttack
# from foolbox.attacks import FGSM,LinfPGD,LinfDeepFoolAttack

from art.attacks.evasion import FastGradientMethod,DeepFool
from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
sys.path.append('../common_code')
import general as g

    
# 画曲线和置信度
def plt_line_conf(data_in):
    y_mean = data_in.mean(axis=0)
#    y_mean = np.median(data_in,axis=0)
    # y_std  = data_in.std(axis=0)
    x      = range(len(y_mean))
    plt.plot(x, y_mean)
    # plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    plt.grid()


# 配置解释器参数
if len(sys.argv)!=4:
    print('Manual Mode !!!')
    model_type  = 'allconv'
    # dir_img     = '../../Dataset/cifar-10/val'
    time_now    = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    img_num     = 30
    # dir_save    = '../saved_tests/20210717_214657'
    # dataset     ='cifar-10'
    # fft_level   = 8
    att_method  = 'FGSM_L2_IDP'
    eps         = 0.5
else:
    print('Terminal Mode !!!')
    model_type  = sys.argv[1]
    # dir_model   = sys.argv[2]
    # dir_img     = sys.argv[2]
    # img_num     = int(sys.argv[2])
    # dir_save    = sys.argv[3]
    # dataset     = sys.argv[5]
    # fft_level   = int(sys.argv[6])
    att_method  = sys.argv[2]
    eps         = float(sys.argv[3])

if ('CW_L2_IDP'==att_method and 0.1!=eps) or ('Deepfool_L2_IDP'==att_method and 0.1!=eps):
    print('[Warning] Skip %s with %s'%(att_method,eps))
    sys.exit(0)
    
flag_norm = 0
flag_imagenet = 0
g.setup_seed(0)
dir_model = '../models/cifar_vanilla_'+model_type+'.pth.tar'
dir_save='../saved_tests/img_shap'
dir_save=dir_save+'/'+model_type+'_'+att_method+'_'+str(eps)
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
                    
model,dataset=g.select_model(model_type, dir_model)
model.eval()

if 'cifar-10'==dataset:
    mean   = g.mean_cifar
    std    = g.std_cifar
    nb_classes = g.nb_classes_cifar
    input_shape=g.input_shape_cifar
    with open(g.dir_feature_cifar) as f:
        features=json.load(f)
    fft_level=g.levels_all_cifar
    dir_img=os.path.join(g.dir_cifar_img,'val')
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

attack,_ = g.select_attack(fmodel, att_method, eps)

# 读取图像列表和标签列表
img_list   = []
label_list = []
for root, dirs, files in os.walk(dir_img):
    if files:
        synset = root.split('/')[-1]
        label  = features[synset]
    for filename in files:
        file_path = os.path.join(root, filename)
        img_list.append(file_path)
        label_list.append(label)

'''Shapley值计算及移除验证'''
# 初始化解释器
# fft_level    = 28
players      = fft_level 
nsamples     = 2 * players + 2048
center_mode  = 1
mask_mode    = 1
my_explainer = wave_shapley_explainer(fft_level=fft_level,nsamples=nsamples,dataset=dataset,model=model,center_mode=center_mode,mask_mode=mask_mode)
epsilons     = [eps]
# Imp          = Imperceptiblility()

test_infos   = pd.DataFrame(columns=['label', 'label_name', 'img_path','fft_level'])
rm_big_all   = np.zeros((len(label_list),fft_level+1))
rm_small_all = np.zeros((len(label_list),fft_level+1))
rm_rand_all  = np.zeros((len(label_list),fft_level+1))
shap_all     = np.zeros((len(label_list),fft_level))
L2_norms     = []
# Img_percs    = []
saved_ix     = 0

pro_idx = np.random.permutation(len(img_list))
img_num_used=min(img_num,len(img_list))
img_used_now=0
fail_pred=0
fail_attack=0

# 加载图像and attack
labels_used=[]
img_used=np.zeros((img_num_used,my_explainer.img_size,my_explainer.img_size,3))
for i in range(img_num_used):
    name = img_list[pro_idx[i]] 
    img  = cv2.imread(name)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (my_explainer.img_size,my_explainer.img_size))
    img_used[i]=img
    labels_used.append(label_list[pro_idx[i]])
img_used=img_used.astype(np.float32)/255.0
img_advs=attack.generate(x=img_used.transpose(0,3,1,2),y=labels_used)   

for img_idx in tqdm(range(img_num_used)):
    # 加载图像
    img  = img_used[img_idx]
    
    # 验证是否预测正确
    pred_label,pred_confs = my_explainer.forward_img(np.expand_dims(img.copy(), axis=0))
    if pred_label != label_list[pro_idx[img_idx]]:# or pred_confs[0,pred_label] < 0.5:
        fail_pred=fail_pred+1
        continue
    
    # 攻击是否成功
    label_in   = [pred_label]
    img_adv    = img_advs[img_idx]
    pred_cln = fmodel.predict(img_adv)
    if pred_cln.argmax(axis=1)[0]==label_in[0]:
        fail_attack=fail_attack+1
        continue

    # 解释原始图像
    img_adv = np.uint8(np.clip(np.round(img_adv.transpose(1,2,0)*255),0,255))
    [label_idx,preds_ori,shap_value_ori] = my_explainer.explain_adv(img, img_adv,label_list[pro_idx[img_idx]])
    
    # pred_synset  = my_explainer.feature_names[str(label_idx)][1]
     
    # 从大到小移除
    players  = shap_value_ori.shape[0]
    rows     = players+1
    shap_idx = np.argsort(-shap_value_ori)
    abla_big = np.ones((rows,players))
    for i in range(rows):
        abla_big[i,shap_idx[list(range(i))]]=0
    
    # 从小到大移除
    shap_idx   = np.argsort(shap_value_ori)
    abla_small = np.ones((rows,players))
    for i in range(rows):
        abla_small[i,shap_idx[list(range(i))]]=0
    
    # 随机移除
    shap_idx  = np.array(random.sample(range(0,players),players))
    abla_rand = np.ones((rows,players))
    for i in range(rows):
        abla_rand[i,shap_idx[list(range(i))]]=0
        
    # 计算置信度
    ablation       = np.vstack((abla_big,abla_small,abla_rand))
    confs_all      = my_explainer.forward_mask_clean(ablation)
    confs_rm_big   = confs_all[0:rows,pred_label]
    confs_rm_small = confs_all[rows:2*rows,pred_label]
    confs_rm_rand  = confs_all[2*rows:,pred_label]
    
    # tu xiang ke jian xing
    diff_mat  = img_adv/255.0-img/255.0
    L2_norm0  = np.linalg.norm(diff_mat[:,:,0].reshape(1,-1),ord=2)
    L2_norm1  = np.linalg.norm(diff_mat[:,:,1].reshape(1,-1),ord=2)
    L2_norm2  = np.linalg.norm(diff_mat[:,:,2].reshape(1,-1),ord=2)
    L2_norm=(L2_norm0+L2_norm1+L2_norm2)/3
    # Img_perc  = Imp.cal_jnd_img(img_adv,img)
    L2_norms.append(L2_norm)
    # Img_percs.append(Img_perc)
    
    
    # 存入数据
    # info_tmp=pd.DataFrame({'label':label_list[pro_idx[img_idx]],
    #               'label_name':pred_synset,
    #               'img_path':name,
    #               'fft_level':fft_level},index=[1])

    # test_infos = test_infos.append(info_tmp,ignore_index=True)
    rm_big_all[saved_ix, :]   = confs_rm_big
    rm_small_all[saved_ix, :] = confs_rm_small
    rm_rand_all[saved_ix, :]  = confs_rm_rand
    shap_all[saved_ix, :]     = shap_value_ori
    
    saved_ix=saved_ix+1
    img_used_now=img_used_now+1

rec_num=img_num_used-fail_pred
att_num=rec_num-fail_attack
att_rat=(att_num+fail_pred)/img_num_used  
print('[IMG used:%d/%d/%d]'%(att_num,rec_num,img_num_used))
# num_log=np.array((att_num,rec_num,img_num_used,att_rat,np.mean(L2_norm)))
# 数据写出
np.savetxt(dir_save+'/rm_big.txt',   rm_big_all)
np.savetxt(dir_save+'/rm_small.txt', rm_small_all)
np.savetxt(dir_save+'/rm_rand.txt',  rm_rand_all)
np.savetxt(dir_save+'/shap_all.txt', shap_all)
# np.savetxt(dir_save+'/nums.txt', num_log)
test_infos.to_csv(dir_save+'/test_infos.csv',sep=",",header=True,index=True,index_label='idx')

'''可视化整个数据集的结果'''
# 移除全零行
del_idx      = np.argwhere(np.all(rm_big_all[..., :] == 0, axis=1))
rm_big_del   = np.delete(rm_big_all, del_idx, axis=0)
rm_small_del = np.delete(rm_small_all, del_idx, axis=0)
rm_rand_del  = np.delete(rm_rand_all, del_idx, axis=0)
shap_del     = np.delete(shap_all, del_idx, axis=0)

# 正则化
if 1 == flag_norm:
    rm_big_show   = rm_big_del/rm_big_del[:,0].reshape(-1,1)
    rm_small_show = rm_small_del/rm_small_del[:,0].reshape(-1,1)
    rm_rand_show  = rm_rand_del/rm_rand_del[:,0].reshape(-1,1)
    shap_show     = shap_del/np.sum(shap_del,axis=1).reshape(-1,1)
else:
    rm_big_show   = rm_big_del
    rm_small_show = rm_small_del
    rm_rand_show  = rm_rand_del
    shap_show     = shap_del

# 绘图并保存
plt.figure()
plt_line_conf(rm_big_show)
plt_line_conf(rm_small_show)
plt_line_conf(rm_rand_show)
plt.savefig(dir_save+'/rm_val.png')
plt.figure()
plt_line_conf(shap_show)
plt.savefig(dir_save+'/shap_all.png')
