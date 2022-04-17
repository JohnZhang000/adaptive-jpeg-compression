import sys
import os
import requests

import matplotlib.pyplot as plt
from PIL import Image

from multiprocessing.spawn import prepare
import torch
import models_mae
import numpy as np

class mae_defender:

    def __init__(self,ckpt='models/mae_pretrain_vit_large.pth'):
        self.ckpt=ckpt
        self.model=self.prepare_model(self.ckpt)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def prepare_model(self,chkpt_dir, arch='mae_vit_large_patch16'):
        # build model
        model = getattr(models_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        # model=torch.nn.DataParallel(model).cuda()
        model=model.cuda()
        return model

    def run_one_image(self,img, model):
        x = torch.tensor((img-self.mean)/self.std)

        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        # run MAE
        loss, y, mask = model(x.float().cuda(), mask_ratio=0.9)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        ret=im_paste[0].numpy()*self.std+self.mean
        return ret

    def defend(self,imgs,labels):
        assert(len(imgs.shape)==4)
        assert(imgs.shape[1]==imgs.shape[2])
        # imgs = np.transpose(imgs,(0,2,3,1))
        imgs_ret=np.zeros_like(imgs)
        for i,img in enumerate(imgs):
            # img_pil=Image.fromarray(np.uint8(img*255))
            # img_pil.save('0.png')

            img_tmp=self.run_one_image(img,self.model)
            imgs_ret[i,...]=img_tmp

            # img_pil=Image.fromarray(np.uint8(img_tmp*255))
            # img_pil.save('1.png')
        
        return imgs_ret,labels


    
    
