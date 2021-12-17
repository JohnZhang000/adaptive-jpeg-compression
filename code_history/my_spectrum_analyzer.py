# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 19:47:03 2021

@author: DELL
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import math
import cv2

class img_spectrum_analyzer:

    # 解释器初始化
    def __init__(self,img_size,mask_mode=1):
        # assert img_shape[0]==img_shape[1]
        self.img_size=img_size
        
        if 1== mask_mode:
            half_size = int(self.img_size/2)
            masks  =np.zeros((self.img_size,self.img_size),np.float64)
        
            r_max  = half_size
            r_list = self.get_radius(r_max,half_size)
     
            r_s_tl = half_size
            r_s_dr = half_size - 1
            for i in range(half_size):
                img=np.zeros((self.img_size,self.img_size))
                r_b_tl=r_s_tl-r_list[i]
                r_b_dr=r_s_dr+r_list[i]
                if half_size-1==i:
                    r_b_tl=0
                    r_b_dr=self.img_size
                cv2.rectangle(img,(r_b_tl,r_b_tl),(r_b_dr,r_b_dr),1,-1)
                if 0!=i:
                    cv2.rectangle(img,(r_s_tl,r_s_tl),(r_s_dr,r_s_dr),0,-1)
                r_s_tl = r_b_tl
                r_s_dr = r_b_dr
                masks  += img*i
            self.fr=masks.astype(np.float32)
            self.max_f=int(self.fr.max()+1)
        elif 2==mask_mode:
            x=np.arange(0,self.img_size,1)
            y=np.arange(0,self.img_size,1)
            [fx,fy]=np.meshgrid(x,y)
            fx=fx-self.img_size/2
            fy=fy-self.img_size/2
            self.fr=np.sqrt(fx**2+fy**2)
            self.max_f=int(round(self.fr.max()))
       
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
    
    def get_spectrum_energy(self, img):
        assert img.shape[-2]==img.shape[-1]
        assert img.shape[-2]==self.img_size
        p=np.fft.fftshift(np.fft.fft2(img,axes=(-2,-1)),axes=(-2,-1))
        p=np.abs(p)
        p=np.log(1+p)
        p=(p-p.min())/(p.max()-p.min()+1e-12)
        
        E=np.zeros(self.max_f)
        for i in range(self.max_f):
            select_idx=(self.fr<=i) &(self.fr>(i-1))*1
            p_i=p*np.expand_dims(select_idx,axis=0)
            E[i]=p_i.sum()/select_idx.sum()/3
        return E.reshape((1,-1)),p
    
    def batch_get_spectrum_energy(self, img_batch):
        assert img_batch.shape[-2]==img_batch.shape[-1]
        Es_list=[]
        Ps_list=[]
        for i in range(img_batch.shape[0]):
            E,p=self.get_spectrum_energy(img_batch[i,...])
            Es_list.append(E)
            Ps_list.append(p)
        Es_np = np.vstack(Es_list)
        Ps_np = np.vstack(Ps_list)
        return Es_np.astype(np.float32),Ps_np.astype(np.float32)
    
    def get_linear_regression(self,x_in,y_in):
        x=x_in#[2:]
        y=y_in#[2:]
        N=len(x)
        sumx = sum(x)
        sumy = sum(y)
        sumx2 = sum(x**2)
        sumy2 = sum(y**2)
        sumxy = sum(x*y)
        A = np.mat([[N,sumx],[sumx,sumx2]])
        b = np.array([sumy,sumxy])
        
        b,k=np.linalg.solve(A, b)
        r=abs(sumy*sumx/N-sumxy)/math.sqrt((sumx2-sumx*sumx/N)*(sumy2-sumy*sumy/N))
        return k,b,r
    
    def get_residual(self,x_in,y_in,k,b):
        x=x_in#[10:]
        y=y_in#[10:]
        
        y_pred=k*x+b
        y_res=y-y_pred
        
        return y_res*10
        
    def get_spectrum_feature(self,img):
        spectrum,_ = self.get_spectrum_energy(img)
        spectrum   = spectrum.reshape((-1))
        feature    = spectrum
        return     feature.reshape((1,-1))
    
    def batch_get_spectrum_feature(self, img_batch):
        assert img_batch.shape[-2]==img_batch.shape[-1]
        Es_list=[]
        for i in range(img_batch.shape[0]):
            E=self.get_spectrum_feature(img_batch[i,...])
            Es_list.append(E)
        Es_np = np.vstack(Es_list)
        return Es_np.astype(np.float32)
        

    
    
if __name__=='__main__':    
    img_pil  = Image.open('n02009912_5714.JPEG').resize((32,32))
    img_np   = np.array(img_pil)
    
    analyzer=img_spectrum_analyzer(32)
    E,p=analyzer.get_spectrum_energy(img_np.transpose(2,0,1))
    f=analyzer.get_spectrum_feature(img_np.transpose(2,0,1))
    
    fig=plt.figure()
    plt.imshow(img_np)
    fig=plt.figure()
    x=np.arange(E.shape[1])+1
    plt.plot(np.log10(x),E[0,:],label='img')
    plt.plot(np.log10(x),np.log10(1/x)+1,label='1/f')       #1/f
    plt.plot(np.log10(x),np.log10(1/(x**2))+1,label='1/f^2')  #1/f^2
    plt.legend()
    fig=plt.figure()
    x=np.arange(0,32,1)
    y=np.arange(0,32,1)
    xx,yy=np.meshgrid(x,y)
    zz_p=p.sum(axis=0)
    ax=fig.add_subplot(1,1,1,projection='3d')
    ax.set_top_view()
    ax.plot_surface(xx,yy,zz_p,rstride=1,cstride=1,cmap='rainbow')
    plt.show()
