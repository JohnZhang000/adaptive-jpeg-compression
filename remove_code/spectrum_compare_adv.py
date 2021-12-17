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
# sys.path.append('../common_code')
import general as g
from img_ploter import img_ploter

def write_xls_col(data,legends,saved_name):
    exl=xlwt.Workbook()
    exl_sheet=exl.add_sheet('data')
    
    for i in range(len(data)):
        exl_sheet.write(0,2*i,'Log Energy')
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

# 输入
# saved_dir  = sys.argv[1]
# model      = [sys.argv[2]]
saved_dir='../saved_tests/img_attack'
model=['vgg16']
att_method=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP',
            'FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP',
            'FGSM_L2_UAP','PGD_L2_UAP','CW_L2_UAP','Deepfool_L2_UAP',]
eps_L2=['0.1','0.5','1.0','10.0','100.0']
eps_Linf=['0.005','0.01','0.1','1.0','10.0']
if 'imagenet' in model[0]:
    spectrums_len=int(g.input_shape_cifar[2]/2)
else:
    spectrums_len=int(g.input_shape_imagenet[2]/2)
    
ploter=img_ploter(fontsize=6)


xlabel          = 'Log Spatial Frequency'
ylabel          = 'Log Power'
spectrums_cln   = np.load(os.path.join(saved_dir,model[0]+'_'+'PGD_L2_IDP'+'_'+eps_L2[0],'spectrum/clean_spectrum.npy'))
x               = np.log10(np.arange(spectrums_cln.shape[1])+1)
flag_fill       = 0
'''
比较所有工况
'''
spectrums=[spectrums_cln]
tick_labels=['natural']
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
            file_name=os.path.join(saved_dir,dir_name,'spectrum/adv_spectrum.npy')
            if os.path.exists(file_name):
                spectrums.append(np.load(file_name))
            else:
                print('Not exist %s'%file_name)
                spectrums.append(np.zeros([1,spectrums_len]))
            tick_labels.append(att_method[j]+'_'+eps[k])
write_xls_col(spectrums,tick_labels,os.path.join(saved_dir,model[i]+'_all.xls'))

