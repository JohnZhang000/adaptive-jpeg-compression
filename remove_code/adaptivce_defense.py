from albumentations import augmentations
from scipy.fftpack import dct, idct, rfft, irfft
import tensorflow as tf
import numpy as np
from albumentations import *
from random import randint, uniform
import PIL
import PIL.Image
from io import BytesIO
import cv2
import random
import pickle
from my_regressor import Net,resnet18,resnet50
import torch
import sys
sys.path.append('../common_code')
import general as g

# This file contains the defense methods compared in the paper.
# The FD algorithm's source code is from:
#   https://github.com/zihaoliu123/Feature-Distillation-DNN-Oriented-JPEG-Compression-Against-Adversarial-Examples/blob/master/utils/feature_distillation.py
# The FD algorithm is refer to the paper:
#   https://arxiv.org/pdf/1803.05787.pdf
# Some of the defense methods' code refering to Anish & Carlini's github: https://github.com/anishathalye/obfuscated-gradients


class adaptive_defender:

    # 解释器初始化
    def __init__(self,table_pkl,dir_model,nb_classes,input_size,pred_batch_size,model_mean_std=None):
        self.grid_size=8
        self.input_size=input_size
        self.pad_size=self.input_size
        self.pred_batch_size=pred_batch_size
        
        
        self.tabel_dict=pickle.load(open(table_pkl,'rb'))
        
        self.model=None
        if not (dir_model is None):
            self.model=resnet50(nb_classes).eval()
            self.model = torch.nn.DataParallel(self.model).cuda()
            checkpoint = torch.load(dir_model)
            self.model.load_state_dict(checkpoint["state_dict"],True)
            
        self.model_mean_std=None
        if model_mean_std:
            self.model_mean_std=np.load(model_mean_std).astype(np.float32)
    
    def compress_with_table(self,input_matrix,table_now):
        output = []
        input_matrix = input_matrix*255

        n = input_matrix.shape[0]
        h = input_matrix.shape[1]
        w = input_matrix.shape[2]
        c = input_matrix.shape[3]
        horizontal_blocks_num = w / self.grid_size
        output2=np.zeros((c,h, w))
        output3=np.zeros((n,3,h, w))
        vertical_blocks_num = h / self.grid_size
        n_block = np.split(input_matrix,n,axis=0)
        for i in range(n):
            c_block = np.split(n_block[i],c,axis =3)
            j=0
            for ch_block in c_block:
                ch_block=ch_block.squeeze()
                vertical_blocks = np.split(ch_block, vertical_blocks_num,axis = 1)
                k=0
                for block_ver in vertical_blocks:
                    block_ver=block_ver.squeeze()
                    hor_blocks = np.split(block_ver,horizontal_blocks_num,axis = 0)
                    m=0
                    for block in hor_blocks:
                        block = np.reshape(block,(self.grid_size,self.grid_size))
                        block = g.dct2(block)
                        # quantization
                        table_quantized = np.matrix.round(np.divide(block, table_now[i,:,:,j]))
                        table_quantized = np.squeeze(np.asarray(table_quantized))
                        # de-quantization
                        table_unquantized = table_quantized*table_now[i,:,:,j]
                        IDCT_table = g.idct2(table_unquantized)
                        if m==0:
                            output=IDCT_table
                        else:
                            output = np.concatenate((output,IDCT_table),axis=0)
                        m=m+1
                    if k==0:
                        output1=output
                    else:
                        output1 = np.concatenate((output1,output),axis=1)
                    k=k+1
                output2[j] = output1
                j=j+1
            output3[i] = output2
        output3 = np.transpose(output3,(0,2,1,3))
        output3 = np.transpose(output3,(0,1,3,2))
        output3 = output3/255
        output3 = np.clip(np.float32(output3),0.0,1.0)
        return output3
    
    def get_adaptive_table(self,imgs,base_imgs):
        imgs_dct=self.img2dct(imgs)
        base_imgs_dct=self.img2dct(base_imgs)        
        diff_dct=imgs_dct-base_imgs_dct#self.base_imgs_dct
        
        tables=np.ones_like(diff_dct)

        threshs=np.ones_like(self.threshs_spec)
        for i in range(len(threshs)):
            block_tmp=base_imgs_dct[:,:,:,i]
            block_tmp_mean=block_tmp.mean(axis=0)
            threshs[i]=block_tmp_mean.max()*self.threshs_spec[i]
        
        for i in range(diff_dct.shape[0]):
            for j in range(diff_dct.shape[3]):
                block_tmp=diff_dct[i,:,:,j]
            
                block_tmp[block_tmp>threshs[j]]=100
                block_tmp[block_tmp<threshs[j]]=1
                tables[i,:,:,j]=block_tmp
        tables[tables==0]=1   
        return tables
           
        
    def scale_table(self,table_now,Q=50):
        # Q =50
        # q_table0=q_table0*Q+np.ones_like(q_table0)
        if Q<=50:
            S=5000/Q
        else:
            S=200-2*Q

        q_table=np.floor((S*table_now+50)/100)
        q_table[q_table==0]=1
        return q_table
    

    
    def get_adaptive_eps(self, imgs):
        imgs_tmp=g.img2dct(imgs)
        if not (self.model_mean_std is None):
            imgs_tmp=(imgs_tmp-self.model_mean_std[...,0:3])/self.model_mean_std[...,3:6]
        imgs_tmp=imgs_tmp.transpose(0,3,1,2)   
        
        eps_list=[]
        batch_size=self.pred_batch_size
        batch_num=int(np.ceil(imgs_tmp.shape[0]/batch_size))
        for i in range(batch_num):
            start_idx=batch_size*i
            end_idx=min(batch_size*(i+1),imgs_tmp.shape[0])
            eps_tmp=self.model(torch.from_numpy(imgs_tmp[start_idx:end_idx,...]).cuda()).detach().cpu().numpy()
            eps_list.append(eps_tmp)
            torch.cuda.empty_cache()
        
        eps_np=np.hstack(eps_list)
        # print(eps_np.shape)
        return eps_np
        
    def img2dct(self,clean_imgs):
        n = clean_imgs.shape[0]
        h = clean_imgs.shape[1]
        w = clean_imgs.shape[2]
        c = clean_imgs.shape[3]
        
        block_dct=np.zeros((n,self.grid_size,self.grid_size,c))

        horizontal_blocks_num = w / self.grid_size
        vertical_blocks_num = h / self.grid_size
        n_block_cln = np.split(clean_imgs,n,axis=0)
        for i in range(n):
            c_block_cln = np.split(n_block_cln[i],c,axis =3)
            for j in range(len(c_block_cln)):
                ch_block_cln=c_block_cln[j].squeeze()
                vertical_blocks_cln = np.split(ch_block_cln, vertical_blocks_num,axis = 1)
                for k in range(len(vertical_blocks_cln)):
                    block_ver_cln=vertical_blocks_cln[k].squeeze()
                    hor_blocks_cln = np.split(block_ver_cln,horizontal_blocks_num,axis = 0)
                    for m in range(len(hor_blocks_cln)):
                        block_cln=hor_blocks_cln[m]
                        block_cln = np.reshape(block_cln,(self.grid_size,self.grid_size))                        
                        block_cln_tmp = np.log(1+np.abs(g.dct2(block_cln)))
                        block_dct[i,:,:,j]=block_dct[i,:,:,j]+block_cln_tmp

        block_dct=block_dct/(horizontal_blocks_num*vertical_blocks_num)
        return block_dct
    
    # def get_cln_dct(self,clean_imgs):
    #     self.base_imgs_dct=np.zeros((self.grid_size,self.grid_size,clean_imgs.shape[3]))
    #     imgs_ycbcr=self.rgb_to_ycbcr(clean_imgs)
    #     imgs_dct=self.img2dct(imgs_ycbcr)
    #     for i in range(imgs_dct.shape[3]):
    #         block_tmp=imgs_dct[:,:,:,i]
    #         block_tmp_mean=block_tmp.mean(axis=0)
    #         self.base_imgs_dct[:,:,i]=block_tmp_mean
    #         self.abs_threshs[i]=block_tmp_mean.max()*self.threshs_spec[i]
            
    def bilnear_table(self,eps):
        tables=np.ones((len(eps),self.grid_size,self.grid_size,3))
        
        keys=list(self.tabel_dict.keys())
        eps=np.clip(eps, keys[0], keys[-1])

        for i in range(len(eps)):
            for j in range(len(keys)-1):
                if keys[j]<=eps[i] and keys[j+1]>=eps[i]:
                    upper_key=keys[j+1]
                    lower_key=keys[j]
                    break
            
            upper_table=self.tabel_dict[upper_key]
            lower_table=self.tabel_dict[lower_key]
            
            table_tmp=lower_table*(upper_key-eps[i])/(upper_key-lower_key)+upper_table*(eps[i]-lower_key)/(upper_key-lower_key)
            tables[i,...]=table_tmp
        return tables
    
    def padresult(self,cleandata):
        
        pad = augmentations.transforms.PadIfNeeded(min_height=self.pad_size, min_width=self.pad_size, border_mode=4)
        paddata = np.ones((cleandata.shape[0],self.pad_size,self.pad_size,3))
        for i in range(paddata.shape[0]):
            paddata[i] = pad(image = cleandata[i])['image']
        return paddata.astype(np.float32())

    def cropresult(self,paddata):
        
        crop = augmentations.crops.transforms.Crop(0,0,self.input_size,self.input_size)
        resultdata = np.ones((paddata.shape[0],self.input_size,self.input_size,3))
        for i in range(resultdata.shape[0]):
            resultdata[i] = crop(image = paddata[i])['image']
        return np.clip(resultdata.astype(np.float32()),0,1)
    
    def defend(self, imgs, labels=None, eps_in=None, flag_flip=1):
        imgs=self.padresult(imgs)
        
        # get eps and tables
        imgs=g.rgb_to_ycbcr(imgs)  
        if eps_in is None:
            eps=self.get_adaptive_eps(imgs)
        else:
            eps=eps_in
        tables=self.bilnear_table(eps)
        
        # adaptive defense
        augeds=self.compress_with_table(imgs,tables)
        augeds=g.ycbcr_to_rgb(augeds)
        augeds = self.cropresult(augeds)
        
        # hor flip
        if flag_flip:
            augeds = np.flip(augeds,2).copy()
        return augeds,labels 
    
def Cal_channel_wise_qtable(clean_imgs,adv_imgs,thresh):
    n = clean_imgs.shape[0]
    h = clean_imgs.shape[1]
    w = clean_imgs.shape[2]
    c = clean_imgs.shape[3]
    num=8
    
    Q0=np.zeros((c,num,num))
    block_diff_all=np.zeros((n,num,num,c))

    block_cln_all=[[] for _ in range(c)]
    block_adv_all=[[] for _ in range(c)]
    block_nums=0


    horizontal_blocks_num = w / num
    vertical_blocks_num = h / num
    n_block_cln = np.split(clean_imgs,n,axis=0)
    n_block_adv = np.split(adv_imgs,n,axis=0)
    for i in range(n):
        c_block_cln = np.split(n_block_cln[i],c,axis =3)
        c_block_adv = np.split(n_block_adv[i],c,axis =3)
        for j in range(len(c_block_cln)):
            ch_block_cln=c_block_cln[j].squeeze()
            ch_block_adv=c_block_adv[j].squeeze()
            vertical_blocks_cln = np.split(ch_block_cln, vertical_blocks_num,axis = 1)
            vertical_blocks_adv = np.split(ch_block_adv, vertical_blocks_num,axis = 1)
            for k in range(len(vertical_blocks_cln)):
                block_ver_cln=vertical_blocks_cln[k].squeeze()
                block_ver_adv=vertical_blocks_adv[k].squeeze()
                hor_blocks_cln = np.split(block_ver_cln,horizontal_blocks_num,axis = 0)
                hor_blocks_adv = np.split(block_ver_adv,horizontal_blocks_num,axis = 0)
                for m in range(len(hor_blocks_cln)):
                    block_nums+=1
                    block_cln=hor_blocks_cln[m]
                    block_adv=hor_blocks_adv[m]
                    block_cln = np.reshape(block_cln,(num,num))
                    block_adv = np.reshape(block_adv,(num,num))
                    
                    # block_cln_dct = g.dct2(block_cln)
                    # block_adv_dct = g.dct2(block_adv)
                    
                    block_cln_tmp = np.log(1+np.abs(g.dct2(block_cln)))
                    block_adv_tmp = np.log(1+np.abs(g.dct2(block_adv)))
                    
                    # block_cln_tmp = np.abs(dct2(block_cln))
                    # block_adv_tmp = np.abs(dct2(block_adv))
                    
                    block_cln_all[j].append(np.expand_dims(block_cln_tmp,axis=0))
                    block_adv_all[j].append(np.expand_dims(block_adv_tmp,axis=0))
                    block_diff = np.abs(block_adv_tmp-block_cln_tmp)
                    # block_diff = block_diff/(block_cln+1e-10)
                    # block_diff = cv2.dct(block_adv-block_cln)
                    # block_diff = np.abs(block_diff/(dct2(block_cln)+1e-10))
                    
                    
                    # block_tmp=np.fft.fft2(block_adv)-np.fft.fft2(block_cln)
                    # block_diff=np.log(1+np.sqrt(block_tmp.real**2+block_tmp.imag**2))                                    
                    
                    # block_cln_all=block_cln_all+block_cln_dct
                    # block_adv_all=block_adv_all+block_adv_dct
                    
                    block_diff_all[i,...,j]=block_diff_all[i,...,j]+block_diff
                    xmax=np.argmax(block_diff)
                    xq=xmax//num
                    yq=xmax%num
                    
                    Q0[j,xq,yq]+=1
              
    # Q=softmax(Q0)
    Q0[Q0==0]=1
    Q=(Q0/Q0.min())
    
    # block_cln_all_mean=np.vstack(block_cln_all).mean(axis=0).reshape([8,8])
    # block_adv_all_mean=np.vstack(block_adv_all).mean(axis=0).reshape([8,8])
    block_diff_all=block_diff_all/(horizontal_blocks_num*vertical_blocks_num)
    block_diff_out=np.ones((num,num,c))
    for j in range(block_diff_all.shape[-1]):
        
        block_cln_tmp=np.vstack(block_cln_all[j])
        block_cln_tmp=block_cln_tmp.mean(axis=0)
        abs_thresh=block_cln_tmp.max()*thresh[j]
        
        block_tmp=block_diff_all[...,j]      
    
        block_tmp[block_tmp>abs_thresh]=100
        block_tmp[block_tmp<abs_thresh]=1
        block_diff_out[...,j]=block_tmp.mean(axis=0)
    return block_diff_out,Q,np.vstack(block_cln_all),np.vstack(block_adv_all)
        