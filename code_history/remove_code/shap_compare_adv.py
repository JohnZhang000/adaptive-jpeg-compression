#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:31:12 2021

@author: dell
"""
import os
import sys
# import matplotlib.pyplot as plt
import numpy as np
# import sys
import xlwt
import shutil
sys.path.append('../common_code')
# from img_ploter import img_ploter
# sys.path.append('../common_code')
import general as g

def read_shap(file_name):
    shap_all=np.loadtxt(file_name)
    del_idx      = np.argwhere(np.all(shap_all[..., :] == 0, axis=1))
    shap_del     = np.delete(shap_all, del_idx, axis=0)
    
    flag_norm = 0
    if 1 == flag_norm:
        shap_show     = shap_del/np.sum(shap_del,axis=1).reshape(-1,1)
    else:
        shap_show     = shap_del
    return shap_show

def read_attr(file_name):
    attr=np.loadtxt(file_name)
    attr[3]=attr[3]*100
    return attr[3:6]

def write_xls_col(data,legends,saved_name):
    exl=xlwt.Workbook()
    exl_sheet=exl.add_sheet('data')
    
    for i in range(len(data)):
        exl_sheet.write(0,2*i,'Importance')
        exl_sheet.write(0,2*i+1,'error')
        
        data_now = data[i].mean(axis=0).astype(np.float64)
        std_now  = data[i].std(axis=0).astype(np.float64)
        names    = legends[i].split('_')
        
        for j in range(len(names)):
            exl_sheet.write(j+1,2*i,names[j])
            
        for j in range(len(data_now)):
            exl_sheet.write(j+len(names)+1,2*i,data_now[j])
            exl_sheet.write(j+len(names)+1,2*i+1,std_now[j])
        
        exl.save('temp.xls')
        shutil.move('temp.xls',saved_name)
    return exl,exl_sheet

def read_shap_single(file_name):
    shap_all     = np.loadtxt(file_name)
    del_idx      = np.argwhere(np.all(shap_all[..., :] == 0, axis=1))
    shap_del     = np.delete(shap_all, del_idx, axis=0)
    
    flag_norm = 1
    if 1 == flag_norm:
        shap_show     = shap_del/np.sum(shap_del,axis=1).reshape(-1,1)
    else:
        shap_show     = shap_del
    return shap_show

def read_shap_batch(saved_dir,method,players):
    model_num=len(method)
    shap_all=np.zeros((model_num,players))
    for i in range(model_num):
        shap_now = read_shap_single(os.path.join(saved_dir,method[i],'shap_all.txt'))
        # shap_all.append(shap_now.mean(axis=0).reshape(1,8))
        shap_all[i,:]=shap_now.mean(axis=0).reshape(1,players)
    return shap_all

def read_pecp_single(file_name):
    nums     = np.loadtxt(file_name)
    pecp     = np.array((nums[4],nums[5]))
    return pecp

def read_pecp_batch(saved_dir,method,players):
    model_num=len(method)
    pecp_num=2
    pecp_all=np.zeros((model_num,pecp_num))
    for i in range(model_num):
        pecp = read_pecp_single(os.path.join(saved_dir,method[i],'nums.txt'))
        pecp_all[i,:]=pecp.reshape(1,pecp_num)
    return pecp_all

# 输入
# saved_dir  = sys.argv[1]
# model      = [sys.argv[2]]
saved_dir='../saved_tests/20211024_124150/shapleys'
model=['resnet50_imagenet']
att_method=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP',
            'FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP',
            'FGSM_L2_UAP','PGD_L2_UAP','CW_L2_UAP','Deepfool_L2_UAP',]

eps_L2=['0.1','0.5','1.0','10.0','100.0']
eps_Linf=['0.005','0.01','0.1','1.0','10.0']
if 'imagenet' in model[0]:
    fft_level=g.levels_all_imagenet
else:
    fft_level=g.levels_all_cifar

'''
比较所有工况
'''
shap_show_all=[]
legends=[]
for i in range(len(model)):
    for j in range(len(att_method)):
        if 'L2' in att_method[j]:
            eps = eps_L2
        else:
            eps=eps_Linf
        if 'CW_L2_IDP' in att_method[j]:
            eps = [eps[0]] 
        if 'Deepfool_L2_IDP' in att_method[j]:
            eps = [eps[0]]
        for k in range(len(eps)):    
            dir_name=model[i]+'_'+att_method[j]+'_'+eps[k]
            file_name=os.path.join(saved_dir,dir_name,'shap_all.txt')
            if os.path.exists(file_name):
                shap_show = read_shap(file_name)
            else:
                print('Not exist %s'%file_name)
                shap_show= np.zeros([1,fft_level])
            # attr=read_attr(os.path.join(saved_dir,dir_name,'nums.txt'))
            shap_show_all.append(shap_show)
            legends.append(dir_name)
        
write_xls_col(shap_show_all,legends,os.path.join(saved_dir,model[i]+'_shap_all.xls'))
