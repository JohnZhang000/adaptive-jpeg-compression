# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from torchvision import datasets, transforms
from scipy.fftpack import dct,idct


from PIL import Image

from pathlib import Path

from timm.models import create_model

import utils
import my_modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_path', default='',type=str, help='input image path')
    parser.add_argument('--save_path', default='',type=str, help='save image path')
    parser.add_argument('--model_path', default='./MAE/results/tmp/20220416_163033_both_npy/checkpoint-4.pth',type=str, help='checkpoint path of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.1, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')

def dct2img(dct_imgs):
    dct_imgs=dct_imgs.numpy()
    assert(4==len(dct_imgs.shape))
    assert(dct_imgs.shape[2]==dct_imgs.shape[3])
    n = dct_imgs.shape[0]
    # h = clean_imgs.shape[1]
    # w = clean_imgs.shape[2]
    c = dct_imgs.shape[1]
    
    block_cln=np.zeros_like(dct_imgs)
    for i in range(n):
        for j in range(c):
            ch_block_cln=dct_imgs[i,j,:,:]                   
            block_cln_tmp = idct2(ch_block_cln)
            block_cln[i,j,:,:]=block_cln_tmp
    return torch.tensor(block_cln)

def ycbcr_to_rgb(imgs):
    imgs=imgs.numpy().transpose(0,2,3,1)
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
    return torch.tensor(imgs_out).permute(0,3,1,2)



def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with open(args.img_path, 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
        print("img path:", args.img_path)

    transforms = DataAugmentationForMAE(args)
    img, bool_masked_pos = transforms(img)
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        img = img[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs = model(img, bool_masked_pos)

        #save original img
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
        ori_img = img * std + mean  # in [0, 1]
        img = ToPILImage()(ori_img[0, :])
        img.save(f"{args.save_path}/ori_img.jpg")

        img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos] = outputs

        #make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        #save reconstruction img
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        img = ToPILImage()(rec_img[0, :].clip(0,0.996))
        img.save(f"{args.save_path}/rec_img.jpg")

        #save random mask img
        img_mask = rec_img * mask
        img = ToPILImage()(img_mask[0, :])
        img.save(f"{args.save_path}/mask_img.jpg")

class mas_defender():
    def __init__(self):
        self.args = get_args()
        self.model = create_model('pretrain_mae_base_patch16_224',
                    pretrained=False,
                    drop_path_rate=0.0,
                    drop_block_rate=None,)
        self.patch_size = self.model.encoder.patch_embed.patch_size
        self.args.window_size = (self.args.input_size // self.patch_size[0], self.args.input_size // self.patch_size[1])
        self.args.patch_size = self.patch_size

        self.model.cuda()
        checkpoint = torch.load(self.args.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.transforms = DataAugmentationForMAE(self.args)


    def defend(self,images,labels):
        assert(len(images.shape)==4)
        assert(images.shape[1]==images.shape[2])
        out_image=[]

        with torch.no_grad():
            for image in images:
                img_pil=Image.fromarray(np.uint8(image*255))

                img, bool_masked_pos = self.transforms(img_pil)
                bool_masked_pos = torch.from_numpy(bool_masked_pos)

                img = img[None, :]
                bool_masked_pos = bool_masked_pos[None, :]
                img = img.cuda()
                bool_masked_pos = bool_masked_pos.cuda().flatten(1).to(torch.bool)
                outputs = self.model(img, bool_masked_pos)

                #save original img
                mean = torch.as_tensor(np.load('spectrum_imagenet_mean.npy')).cuda()[None, :]   #zhang
                std = torch.as_tensor(np.load('spectrum_imagenet_std.npy')).cuda()[None, :]   #zhang
                # mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).cuda()[None, :, None, None]
                # std = torch.as_tensor(IMAGENET_DEFAULT_STD).cuda()[None, :, None, None]
                # mean = torch.as_tensor((0.04008908,0.00953541,0.00794689)).cuda()[None, :, None, None]
                # std = torch.as_tensor((0.03745347,0.01009368,0.00836723)).cuda()[None, :, None, None]
                ori_img = img * std + mean  # in [0, 1]
                # img = ToPILImage()(ori_img[0, :])
                # img.save(f"{args.save_path}/ori_img.jpg")

                img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=self.patch_size[0], p2=self.patch_size[0])
                img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
                img_patch[bool_masked_pos] = outputs

                #make mask
                mask = torch.ones_like(img_patch)
                mask[bool_masked_pos] = 0
                mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
                mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size[0], p2=self.patch_size[1], h=14, w=14)

                #save reconstruction img
                rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
                # Notice: To visualize the reconstruction image, we add the predict and the original mean and var of each patch. Issue #40
                rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
                rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size[0], p2=self.patch_size[1], h=14, w=14)
                # img = ToPILImage()(rec_img[0, :].clip(0,0.996))
                # img.save(f"{args.save_path}/rec_img.jpg")

                #save random mask img
                # img_mask = rec_img * mask
                # img = ToPILImage()(img_mask[0, :])
                # img.save(f"{args.save_path}/mask_img.jpg")

                # rec_img= ori_img[None,:]
                rec_img= dct2img(rec_img.cpu())  #zhang
                rec_img= Image.fromarray(np.transpose(np.uint8(rec_img.squeeze().numpy()*255),(1,2,0)),mode='YCbCr')
                rec_img=rec_img.convert('RGB')
                rec_img= np.array(rec_img)
                out_image.append(rec_img[None,:])
        out_image=np.vstack(out_image)
        # out_image=out_image
        return out_image,labels

if __name__ == '__main__':
    opts = get_args()
    main(opts)
