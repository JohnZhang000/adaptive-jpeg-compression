# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 08:39:46 2020

@author: DELL
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' #只显示警告和报错
import numpy as np
# import pywt
# import copy
#import tensorflow as tf
#from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.backend.tensorflow_backend import set_session
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from cupy.core.dlpack import toDlpack
# from cupy.core.dlpack import fromDlpack
# from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import shap
import json
#from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
# from skimage.segmentation import slic, mark_boundaries
# from matplotlib.colors import LinearSegmentedColormap
from decimal import *
import time
# from tqdm import tqdm
from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD
# from foolbox.utils import samples
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' #只显示警告和报错
# from numba import jit
import cupy as cp
# import transplant
# matlab=transplant.Matlab(jvm=False,desktop=False)

#设置GPU显存动态增长
#gpu_fraction = 0.1
#if gpu_fraction is not None:
#    assert gpu_fraction >0 and gpu_fraction < 1, "invalid gpu_fraction={}".format(gpu_fraction)
#    sessconfig = tf.compat.v1.ConfigProto()
#    # sessconfig.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
#    sessconfig.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#    session = tf.compat.v1.Session(config=sessconfig)
#    tf.compat.v1.keras.backend.set_session(session)
    
class wave_shapley_explainer:

    # 解释器初始化
    # def __init__(self,fft_level,seg_num,nsamples,model,center_mode=1,mask_mode=1):
    def __init__(self,fft_level,nsamples,model,dataset='imagenet', center_mode=1,mask_mode=1):
        print(dataset)
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
        self.trans_adv=dict(mean=mean_now, std=std_now,axis=-3)
        half_size=int(self.img_size/2)
        assert (half_size % fft_level) == 0, '112 should be divided by fft_level with no remainder!'
        self.fft_level  = fft_level            #分解级别
        # self.seg_num    = seg_num              #分割数量
        self.players    = fft_level #* seg_num  #小波成分数目
        self.nsamples   = nsamples             #迭代次数
        self.model      = model                #待解释模型
        # self.segment    = np.zeros((224,224))  #分割mask
        self.background = 0                    #背景颜色
        self.center_mode= 1                    #是否移频至中心
        self.masks_full=cp.ones((self.img_size,self.img_size,3))
        self.batch_num=32

        self.masks  =np.zeros((self.fft_level,self.img_size,self.img_size,3),np.uint8)
        center = half_size        
        if 1== mask_mode:
            # 创建矩形环mask
            
            r_max  = half_size
            # r_list = self.get_radius(r_max,self.fft_level)
            
            # if self.fft_level>10:
            #     ins = 10
            # else:
            #     ins = 0
            ins = 0
            r_list = self.get_radius(r_max-ins,self.fft_level-ins)
            for i in range(ins):
                r_list.insert(0,1)
    
    
            r_s_tl = center
            r_s_dr = center - 1
            for i in range(self.fft_level):
                img=np.zeros((self.img_size,self.img_size))
                r_b_tl=r_s_tl-r_list[i]
                r_b_dr=r_s_dr+r_list[i]
                if self.fft_level-1==i:
                    r_b_tl=0
                    r_b_dr=self.img_size
                cv2.rectangle(img,(r_b_tl,r_b_tl),(r_b_dr,r_b_dr),1,-1)
                if 0!=i:
                    cv2.rectangle(img,(r_s_tl,r_s_tl),(r_s_dr,r_s_dr),0,-1)
                r_s_tl = r_b_tl
                r_s_dr = r_b_dr
                self.masks[i,:,:,0]=img
                self.masks[i,:,:,1]=img
                self.masks[i,:,:,2]=img
                
        elif 2==mask_mode:
            x=np.arange(0,self.img_size,1)
            y=np.arange(0,self.img_size,1)
            [fx,fy]=np.meshgrid(x,y)
            fx=fx-center
            fy=fy-center
            img=np.sqrt(fx**2+fy**2)
            
            rad_gap=(img.max()+1e-5)/self.fft_level
            for i in range(self.fft_level):
                img_tmp=(img<(i+1)*rad_gap) & (img>=i*rad_gap) *1
                self.masks[i,:,:,0]=img_tmp
                self.masks[i,:,:,1]=img_tmp
                self.masks[i,:,:,2]=img_tmp
            
            

            
        self.masks_cp = cp.asanyarray(self.masks)


        # # 创建颜色映射表
        # colors = []
        # for l in np.linspace(1,0,100):
        #     colors.append((245/255,39/255,87/255,l))
        # for l in np.linspace(0,1,100):
        #     colors.append((24/255,196/255,93/255,l))
        # self.cm = LinearSegmentedColormap.from_list("shap", colors)
            
        #加载标签   
        with open("../models/cifar-10_class_to_idx.json") as f:
            self.feature_names=json.load(f)
            
    # def segment_init(self,img):
    #     self.segment = slic(img, n_segments=self.seg_num)
 
    def model_pred(self,model,img):
        tran=self.trans
        # print(img.shape)
        img_cp_t=from_dlpack(toDlpack(img)).to(torch.float32)
        if img_cp_t.max()>10:
            img_cp_t=img_cp_t.div(255.0)
        if 3==len(img_cp_t.shape):
            if not img_cp_t.shape[-1]==img_cp_t.shape[-2]:
                img_cp_tz=img_cp_t.permute(2,0,1)#transpose(0,1).transpose(0,2)#permute(2,0,1)
            img_t = tran(img_cp_tz).unsqueeze(0)	# 将图片转化成tensor
        elif 4==len(img_cp_t.shape):
            if not img_cp_t.shape[-1]==img_cp_t.shape[-2]:
                img_cp_t=img_cp_t.permute(0,3,1,2)
            img_t = torch.zeros([img_cp_t.shape[0],img_cp_t.shape[1],img_cp_t.shape[2],img_cp_t.shape[3]]).cuda()
            # print(img_cp_t.shape)
            for i in range(img_cp_t.shape[0]):
                # a=tran(img_cp_t[i,:,:,:])
                img_t[i,:,:,:]=tran(img_cp_t[i,:,:,:])
        img_t=img_t.cuda().to(torch.float32)
        # batch_num=704
        batch_num=self.batch_num
        test_round = int(np.ceil(img_t.shape[0]/batch_num))
        pred_all   = torch.zeros([img_t.shape[0],self.num_classes]).cuda()
        # m = nn.Softmax(dim=1)
        with torch.no_grad():
            for i in range(test_round):
                start_idx = i*batch_num
                end_idx   = min((i+1)*batch_num,img_t.shape[0])
                img_tmp   = img_t[start_idx:end_idx,:,:,:]
                pred_tmp  = model(img_tmp)
                pred_all[start_idx:end_idx,:]=pred_tmp
        pred_all=F.softmax(pred_all,dim=1)
        torch.cuda.empty_cache()
        return pred_all.detach().cpu().numpy()
    
    def batch_pred(self, zs, dft):
        batch_num=self.batch_num
        test_round = int(np.ceil(zs.shape[0]/batch_num))
        pred_all   = np.zeros([zs.shape[0],self.num_classes])
        for i in range(test_round):
            start_idx = i*batch_num
            end_idx   = min((i+1)*batch_num,zs.shape[0])
            zs_tmp   = zs[start_idx:end_idx,:]   
            pred_tmp  = self.model_pred(self.model,self.mask_clean(zs_tmp,self.dft))
            pred_all[start_idx:end_idx,:]=pred_tmp
        return pred_all
        
    # def batch_pred_adv(self, zs, dft):
    #     batch_num=self.batch_num
    #     test_round = int(np.ceil(zs.shape[0]/batch_num))
    #     pred_all   = np.zeros([zs.shape[0],self.num_classes])
    #     for i in range(test_round):
    #         start_idx = i*batch_num
    #         end_idx   = min((i+1)*batch_num,zs.shape[0])
    #         zs_tmp   = zs[start_idx:end_idx,:]   
    #         pred_tmp  = self.model_pred_adv(self.model,self.mask_adv(zs_tmp,self.dft))
    #         pred_all[start_idx:end_idx,:]=pred_tmp
    #     return pred_all

    
    def get_radius(self, r_max, n):
        """
        把整数均分为若干整数
        
        r_max 最大半径
        n     划分份数
        """
        assert n > 0
        quotient = int(r_max / n)
        remainder = r_max % n
        if remainder > 0:
            return [quotient] * (n - remainder) + [quotient + 1] * remainder
        if remainder < 0:
            return [quotient - 1] * -remainder + [quotient] * (n + remainder)
        return [quotient] * n
    
    def my_fft_single(self,img):
#        dft = np.zeros(img.shape,dtype=complex)
        dft  = cp.fft.fft2(img,axes=(0,1))
        if 1==self.center_mode:
            dft = cp.fft.fftshift(dft,axes=(0,1)) 
        return dft
            
    def my_ifft_single(self,dft):
        if 1==self.center_mode:
            dft = cp.fft.ifftshift(dft,axes=(1,2)) 
        img_tmp = cp.fft.ifft2(dft,axes=(1,2))
        img     = img_tmp.real        
        return img

    def mask_clean(self, zs, dft):
        mask_tmp = cp.outer(zs,self.masks_full).reshape((zs.shape[0],zs.shape[1],self.masks_full.shape[0],self.masks_full.shape[1],self.masks_full.shape[2]))
        mask_tmp = mask_tmp * self.masks_cp
        mask_now = mask_tmp.sum(axis=1)
        
        dft_now  = mask_now*dft
        idft_now = self.my_ifft_single(dft_now)
        if 1 == self.mode_adv:
            idft_now = self.img_ori+idft_now
        img      = cp.clip(cp.around(idft_now),0,255)
        mask_tmp = None
        mask_now = None
        dft_now  = None
        idft_now = None
        return img#cp.asnumpy(out)
        
    # def mask_adv(self, zs, dft):
    #     mask_tmp = cp.outer(zs,self.masks_full).reshape((zs.shape[0],zs.shape[1],self.masks_full.shape[0],self.masks_full.shape[1],self.masks_full.shape[2]))
    #     mask_tmp = mask_tmp * self.masks_cp
    #     mask_now = mask_tmp.sum(axis=1)
        
    #     dft_now  = mask_now*dft
    #     idft_now = self.my_ifft_single(dft_now)
    #     idft_now+now
    #     img      = cp.clip(cp.around(idft_now),0,255)
    #     mask_tmp = None
    #     mask_now = None
    #     dft_now  = None
    #     idft_now = None
    #     return img#cp.asnumpy(out)
    
    def forward_img(self,img):
        img_cp=cp.asarray(img)
        preds = self.model_pred(self.model,img_cp)
        top_preds = np.argsort(-preds)
        label = top_preds[0][0]
        return label,preds
    
    def forward_mask_clean(self,z):
        z_cp  = cp.asarray(z)
        preds = self.batch_pred(z_cp,self.dft)
        if 0 != self.mode_adv:
            preds = self.preds - preds
        z_cp  = None
        return preds

    def explain_clean(self,img,select_ind=-1):
        assert(img.shape[0]==img.shape[1])
        # 原始图像小波分解
        self.mode_adv = 0
        self.img_ori  = cp.asarray(img)
        self.dft = self.my_fft_single(self.img_ori)
        
        # 获得原始样本类别
        preds = self.model_pred(self.model,self.img_ori)
        self.preds = preds
        top_preds = np.argsort(-preds)
        max_pred_ind = top_preds[0][0]
        if -1 == select_ind:
            select_ind = max_pred_ind

        # 沙普利值计算
        explainer   = shap.KernelExplainer(self.forward_mask_clean, np.zeros((1,self.players)))
        shap_values = explainer.shap_values(np.ones((1,self.players)), nsamples=self.nsamples, silent=True)
        shap_value  = shap_values[select_ind][0]

        return select_ind,preds,shap_value
        
    # def forward_mask_adv(self,z):
    #     z_cp  = cp.asarray(z)
    #     preds = self.batch_pred_adv(z_cp,self.dft)
    #     z_cp  = None
    #     return preds

    def explain_diff(self,img,img_adv,select_ind=-1):
        assert(img.shape[0]==img.shape[1])
        assert(img_adv.shape[0]==img_adv.shape[1])
        # 原始图像小波分解
        self.mode_adv = 1
        self.img_ori  = cp.asarray(img).astype(np.float32)
        self.img_adv  = cp.asarray(img_adv).astype(np.float32)
        img_diff=self.img_adv-self.img_ori
        self.dft = self.my_fft_single(img_diff)
        
        # 获得原始样本类别
        preds = self.model_pred(self.model,self.img_ori)
        self.preds = preds
        top_preds = np.argsort(-preds)
        max_pred_ind = top_preds[0][0]
        if -1 == select_ind:
            select_ind = max_pred_ind

        # 沙普利值计算
        explainer   = shap.KernelExplainer(self.forward_mask_clean, np.zeros((1,self.players)))
        shap_values = explainer.shap_values(np.ones((1,self.players)), nsamples=self.nsamples, silent=True)
        shap_value  = shap_values[select_ind][0]

        return select_ind,preds,shap_value
    
    def explain_adv(self,img,img_adv,select_ind=-1):
        assert(img.shape[0]==img.shape[1])
        # 原始图像小波分解
        self.mode_adv = 2
        self.img_ori  = cp.asarray(img).astype(np.float32)
        self.img_adv  = cp.asarray(img_adv).astype(np.float32)
        self.dft = self.my_fft_single(self.img_adv)
        
        # 获得原始样本类别
        preds = self.model_pred(self.model,self.img_ori)
        self.preds = preds
        top_preds = np.argsort(-preds)
        max_pred_ind = top_preds[0][0]
        if -1 == select_ind:
            select_ind = max_pred_ind

        # 沙普利值计算
        explainer   = shap.KernelExplainer(self.forward_mask_clean, np.zeros((1,self.players)))
        shap_values = explainer.shap_values(np.ones((1,self.players)), nsamples=self.nsamples, silent=True)
        shap_value  = shap_values[select_ind][0]

        return select_ind,preds,shap_value

    def plot_bar_img_shap(self,img,shapley,pred_name,pred_conf):
        img_tmp = img
        if 4==len(img_tmp.shape):
            img_tmp=img_tmp.squeeze(0)
        plt.figure()
        plt.subplot(121)
        plt.imshow(np.int32(img_tmp))
        plt.title(pred_name+' '+str(pred_conf))
        plt.axis('off')
        # plt.show()
    
        plt.subplot(122)
        plt.bar(np.arange(self.players),shapley, alpha=0.5, width=0.3, color='yellow', edgecolor='red',)
        plt.title(pred_name+' '+str(sum(shapley)))
        # plt.show()
        
    def plot_ablation_comp(self,img_ori,img_abl,pred_name_ori,pred_conf_ori,pred_name_abl,pred_conf_abl):
        img_ori_tmp=img_ori
        img_abl_tmp=img_abl
        
        if 4==len(img_ori_tmp.shape):
            img_ori_tmp=img_ori_tmp.squeeze(0)
        if 4==len(img_abl_tmp.shape):
            img_abl_tmp=img_abl_tmp.squeeze(0)
            
        # print(img_abl_tmp.shape)
        plt.figure()
        plt.subplot(131)
        plt.imshow(np.int32(img_ori_tmp))
        plt.axis('off')
        plt.title(pred_name_ori+' '+str(pred_conf_ori))
        # plt.show()
        plt.subplot(132)
        plt.imshow(np.int32(img_abl_tmp))
        plt.axis('off')
        plt.title(pred_name_abl+' '+str(pred_conf_abl))
        # plt.show()
        plt.subplot(133)
        plt.imshow(np.int32((img_ori_tmp-img_abl_tmp)))
        plt.axis('off')
        plt.title(pred_name_ori+' '+str(pred_conf_ori-pred_conf_abl))
        # plt.show()
        
    # def fill_segmentation(self,values, segmentation):
    #     out = np.zeros(segmentation.shape)
    #     for i in range(len(values)):
    #         out[segmentation == i] = values[i]
    #     return out
    
    def plot_mask_img_shap(self,img,shapley):
        img_level     = np.zeros((self.fft_level,self.img_size,self.img_size,3),dtype=np.uint8)      #各频域图像
        shapley_level = np.reshape(shapley,[self.fft_level,-1]) #各频域沙普利值
        dft           = self.my_fft_single(img)                 #频域成分
        
        for i in range(self.fft_level):
            masked_dft = dft * self.masks[i,:,:,:]
            idft       = self.my_ifft_single(masked_dft)
            idft_img   = np.uint8(np.clip(np.round(idft),0,255))
            img_level[i,:,:,:] = idft_img
            # img_level[i,:,:]=cv2.cvtColor(idft_img,cv2.COLOR_BGR2GRAY)
        
        # 显示各频域重要性
        img_per_col = 5
        row = int(np.ceil(self.fft_level/5))
        col = img_per_col
        fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(12,4))
        max_val = shapley.max()
        
        if self.fft_level<=img_per_col:
            for i in range(self.fft_level):
                m = self.fill_segmentation(shapley_level[i,:], self.segment)
                img = img_level[i,:,:,:]
                axes[i].set_title('fft_level:'+str(i))
                # axes[i].imshow(img, alpha=0.3)
                im = axes[i].imshow(m, cmap=self.cm, vmin=-max_val, vmax=max_val)
                axes[i].axis('off')
        else:
            for i in range(self.fft_level):
                m = self.fill_segmentation(shapley_level[i,:], self.segment)
                img = img_level[i,:,:,:]
                axes[i//img_per_col,i%img_per_col].set_title('fft_level:'+str(i))
                # axes[i//img_per_col,i%img_per_col].imshow(img, alpha=1)
                im = axes[i//img_per_col,i%img_per_col].imshow(m, cmap=self.cm, vmin=-max_val, vmax=max_val)
                axes[i//img_per_col,i%img_per_col].axis('off')
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
        cb.outline.set_visible(False)
        
        # 单独显示各频域图像
        show_levels = 5
        shapley_level_sum = shapley_level.sum(axis=1)
        idxs = np.argsort(-shapley_level_sum)
        fig, axes = plt.subplots(nrows=1, ncols=show_levels, figsize=(12,4))
        multiply = 1
        for i in range(show_levels):
            img = img_level[idxs[i],:,:,:]
            if img.max()<25:
                multiply = 10
            elif img.max()<50:
                multiply = 5
            elif img.max()<100:
                multiply = 2
            axes[i].imshow(img*multiply)
            axes[i].set_title('fft_level:'+str(idxs[i])+' '+str(Decimal(shapley_level_sum[idxs[i]]).quantize(Decimal('0.0000')))+'\nmultiply:'+str(multiply))
            
        
        

if __name__=='__main__':
    
    # 配置解释器参数
    fft_level   = 28
    model = models.resnet50(pretrained=True).cuda()
    model.eval()

    # 加载图像
    file = "ILSVRC2012_val_00009034.JPEG"
    img=cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(img, (224,224)) 
    
    # 初始化解释器
    center_mode  = 0 
    players      = fft_level
    nsamples     = 2 * players + 2048
    mask_mode    = 2
    my_explainer = wave_shapley_explainer(fft_level,nsamples,model,center_mode=center_mode,mask_mode=mask_mode)

    # 解释原始图像
    start = time.process_time()
    [label_idx,preds_ori,shap_value_ori] = my_explainer.explain_clean(img_orig)
    pred_name_ori  = my_explainer.feature_names[str(label_idx)][1]
    pred_conf_ori  = preds_ori[0,label_idx]
    my_explainer.plot_bar_img_shap(img_orig,shap_value_ori,pred_name_ori,pred_conf_ori)
    # my_explainer.plot_mask_img_shap(img_orig,shap_value_ori)
    end = time.process_time()
    print('time used :%d'%(end-start))

    # 烧蚀原图验证
    ablation_list=[0,1]
    zs=np.ones((1,players))
    zs[0,ablation_list]=0
    z_cp=cp.asarray(zs)
    image_abl  = my_explainer.mask_clean(z_cp, my_explainer.dft)
    
    preds_ori = my_explainer.model_pred(my_explainer.model,cp.asarray(img_orig))
    top_preds = np.argsort(-preds_ori)
    max_pred_ind = top_preds[0][0]
    pred_name  = my_explainer.feature_names[str(max_pred_ind)][1]
    pred_conf  = preds_ori[0,max_pred_ind]

    preds_abl = my_explainer.model_pred(my_explainer.model,cp.asarray(image_abl.squeeze(0)))
    pred_conf_abl = preds_abl[0,max_pred_ind]
    my_explainer.plot_ablation_comp(img_orig,cp.asnumpy(image_abl),pred_name,pred_conf,pred_name,pred_conf_abl)
    
    # 生成对抗样本
    img_adv_in=torch.from_numpy(np.float32(img_orig/255.0)).permute(2,0,1).unsqueeze(0).cuda()
    label_in=torch.tensor([label_idx]).cuda()
    
    fmodel = PyTorchModel(model, bounds=(0, 1),preprocessing=my_explainer.trans_adv)
    attack = LinfPGD()
    epsilons = [0.01]
    raw_advs, clipped_advs, success = attack(fmodel, img_adv_in, label_in, epsilons=epsilons)
    print(success)
    img_adv = np.uint8(np.clip(np.round(raw_advs[0].squeeze(0).permute(1,2,0).cpu().numpy()*255),0,255))
    [label_idx,preds_ori,shap_value_ori] = my_explainer.explain_adv(img_orig,img_adv)
    my_explainer.plot_bar_img_shap(img_orig,shap_value_ori,pred_name_ori,pred_conf_ori)

