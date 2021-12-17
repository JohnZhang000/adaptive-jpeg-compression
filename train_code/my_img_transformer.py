# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:33:25 2021

@author: DELL
"""

from PIL import Image
import numpy as np
import cv2
import os
import sys
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from skimage.util import view_as_blocks

class img_transformer:

    # 解释器初始化
    def __init__(self,fft_level,rpl0=0,start_level=4,img_size=32):
        # print('--------------------------')
        self.fft_level          = fft_level            #分解级别
        self.start_level        = start_level          #tihuanweizhi 
        self.rpl0               = rpl0                 #shifoutihuan0
        self.img_size           = img_size
        self.half_size          = int(self.img_size/2)
        self.masks              = self.create_mask(self.fft_level,self.rpl0,self.start_level)
        self.masks_tc           = np.transpose(self.masks,(2,0,1))
        self.jpg_quality        = 50#23
        self.dir_tmp            = 'temp'
        self.pca_components     = 4
        self.pca_block_shape    = (4,4)


    def create_mask(self,fft_level,rpl0, start_level): 
        assert(self.half_size % fft_level)==0, '112 should be divided by fft_level with no remainder!'    
        masks  =np.zeros((fft_level,self.img_size,self.img_size,3))
        offset = int(self.half_size/fft_level)
        r_list = [offset]*fft_level
        r_s_tl = self.half_size
        r_s_dr = self.half_size -1
        for i in range(fft_level):
            img=np.zeros((self.img_size,self.img_size))
            if i >= rpl0 and i < start_level:
                r_b_tl=r_s_tl-r_list[i]
                r_b_dr=r_s_dr+r_list[i]
                if fft_level-1==i:
                    r_b_tl=0
                    r_b_dr=self.img_size
                cv2.rectangle(img,(r_b_tl,r_b_tl),(r_b_dr,r_b_dr),1,-1)
                if 0!=i:
                    cv2.rectangle(img,(r_s_tl,r_s_tl),(r_s_dr,r_s_dr),0,-1)
                r_s_tl = r_b_tl
                r_s_dr = r_b_dr
            masks[i,:,:,0]=img
            masks[i,:,:,1]=img
            masks[i,:,:,2]=img
        masks_ret=masks.sum(axis=0)
        return masks_ret
        
    def img_transform(self,img_pil): 
        img_in            = np.array(img_pil.convert('RGB').resize((self.img_size,self.img_size)))
        img_fft           = np.fft.fft2(img_in,axes=(0,1))
        img_ifft1         = np.fft.fftshift(img_fft,axes=(0,1)) 
        masked_ifft       = self.masks *img_ifft1
        masked_ifft_shift = np.fft.ifftshift(masked_ifft,axes=(0,1)) 
        img_tmp           = np.fft.ifft2(masked_ifft_shift,axes=(0,1))
        idfts             = np.clip(img_tmp.real,0,255)
        img_out           = Image.fromarray(idfts.astype('uint8'))
        return img_out
                
    def img_transform_tc(self,img_tc): 
        assert img_tc.shape[-2]==img_tc.shape[-1]
        img_in            = img_tc
        flag_tc           = 0
        if torch.is_tensor(img_tc):
            img_in        = img_tc.cpu().numpy()
            flag_tc       = 1            
        img_fft           = np.fft.fft2(img_in,axes=(-2,-1))
        img_ifft1         = np.fft.fftshift(img_fft,axes=(-2,-1)) 
        masked_ifft       = img_ifft1 * self.masks_tc
        masked_ifft_shift = np.fft.ifftshift(masked_ifft,axes=(-2,-1)) 
        img_tmp           = np.fft.ifft2(masked_ifft_shift,axes=(-2,-1))
        idfts             = np.clip(img_tmp.real,0,1).astype(np.float32)
        img_out           = idfts
        if flag_tc:
            img_out           = torch.from_numpy(idfts).cuda()
        return img_out

    
    def jpg_compression_tc(self,img_tc): 
        dir_tmp           = self.dir_tmp
        if not os.path.exists(dir_tmp):
                os.makedirs(dir_tmp)
        loader=transforms.ToTensor()
        unloader = transforms.ToPILImage()
        
        img_in            = img_tc.cpu()
        for i in range(img_in.shape[0]):
            img     = img_in[i].squeeze(0)
            img_pil = unloader(img)
            img_pil.save(os.path.join(dir_tmp,str(i)+".JPEG"),quality=self.jpg_quality)
        
        img_out           = torch.zeros_like(img_in)
        for i in range(img_out.shape[0]):
            img     = Image.open(os.path.join(dir_tmp,str(i)+".JPEG"))
            img_tc  = loader(img)
            img_out[i,...]=loader(img)
            
        return img_out.cuda()
    
    def pca_whole_tc(self,img_tc): 
        img_in            = img_tc.cpu().numpy()
        img_out           = np.zeros_like(img_in)
        pca = PCA(n_components = self.pca_components)

        for i in range(img_in.shape[0]):
            for j in range(img_in.shape[1]):
                img_onecolor          = img_in[i,j,:,:]
                pca.fit(img_onecolor)
                onecolor_pca          = pca.fit_transform(img_onecolor)
                img_onecolor_restored = pca.inverse_transform(onecolor_pca)
                img_out[i,j,:,:]      = np.clip(img_onecolor_restored,0,1)
            
        return torch.from_numpy(img_out).cuda()
    
    def pca_block_tc(self,img_tc): 
        img_in            = img_tc.cpu().numpy()
        img_out           = np.zeros_like(img_in)
        pca = PCA(n_components = self.pca_components)
        block_shape       = self.pca_block_shape
        img_size          = self.img_size

        for i in range(img_in.shape[0]):
            for j in range(img_in.shape[1]):
                img_onecolor = img_in[i,j,:,:]
                img_onecolor_blocks          = view_as_blocks(img_onecolor, block_shape)
                img_onecolor_restored_blocks = np.empty_like(img_onecolor_blocks)
                
                shape = img_onecolor_blocks.shape
                for (k, l) in np.ndindex((shape[2], shape[3])):
                    block = img_onecolor_blocks[:,:,k,l]
                    pca.fit(block)
                    block_pca = pca.fit_transform(block)
                    img_onecolor_restored_blocks[:,:,k,l] = pca.inverse_transform(block_pca)
                x = img_onecolor_restored_blocks.transpose(0,2,1,3).reshape(img_size,img_size)
                img_out[i,j,:,:]=np.clip(x,0,1)
            
        return torch.from_numpy(img_out).cuda()
        
        
if __name__=='__main__':    

    fft_level   = 8
    rpl0        = 0
    start_level = 4
    transformer = img_transformer(fft_level,rpl0,start_level)
    
    src_name = 'a.JPEG'
    img_pil  = Image.open(src_name)
    img_tsf  = transformer.img_transform(img_pil)
    img_tsf.save(src_name.replace('.JPEG','_tsf.JPEG'))
    
    img_np   = np.array(img_pil.resize((32,32)))
    img_np   = np.expand_dims(img_np, axis=0)
    img_np   = np.repeat(img_np, 5, axis=0)
    img_np   = img_np.astype(np.float32)/255.0
    img_tc   = torch.from_numpy(img_np).permute(0,3,1,2)
    
    img_jpg  = transformer.pca_block_tc(img_tc)
    img_jpg_np=img_jpg.cpu().numpy()


