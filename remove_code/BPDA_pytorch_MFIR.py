import os
from secrets import choice
import shutil
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize,one_hot
sys.path.append('../../../common_code')
sys.path.append('../common_code')
import general as g
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from defense import tctensorGD
import logging

import defense_ago
# defense_ago.input_size=224
# defense_ago.pad_size=229
from defense import defend_webpf_wrap,defend_webpf_my_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
from defense import defend_my_webpf,defend_jpeg_wrap,defend_bdr_wrap,defend_gaua_wrap,tctensorGD_warp
from defense_ago import defend_FD_ago_warp,defend_my_fd_ago
from fd_jpeg import fddnn_defend
from adaptivce_defense import adaptive_defender

from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin

import argparse
import cv2
from torch.nn.functional import softmax


class MFIR_defender:

    # 解释器初始化
    def __init__(self,model,mean,std,input_size):
        c,h,w=input_size
        self.model=model
        self.mean=mean
        self.std=std
        if 32==h:
            self.base=2
            self.std1=2
            self.std2=2
        else:
            self.base=20
            self.std1=20
            self.std2=20
        self.transform=transforms.Compose([transforms.RandomResizedCrop((h,w)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean,self.std),])
    
    def pred(self,img):
        img=self.transform(img)
        logits=self.model(img.unsqueeze(0).cuda())
        logits=softmax(logits.detach().cpu(),dim=1)
        return logits.numpy()

    def defend_single(self,img):
        img=np.uint8(img*255)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.bilateralFilter(img,3,self.std1,self.std2)
        img=cv2.bilateralFilter(img,5,self.std1,self.std2)
        logits=[]
        for i in range(10):
            rb=np.random.random()*self.base-self.base/2
            rc=np.random.random()*20-10
            img_tmp=cv2.bilateralFilter(img,self.base-1,1,self.base*i+rb)
            img_tmp=Image.fromarray(cv2.cvtColor(img_tmp,cv2.COLOR_BGR2RGB))
            img_tmp=img_tmp.rotate(rc,expand=1.1)
            logit=self.pred(img_tmp)
            logits.append(logit)
        logits=np.vstack(logits)
        pred=np.argmax(logits.mean(axis=0))
        return pred

    def defend_batch(self,imgs,labels):
        correct=0
        img_num=imgs.shape[0]
        for i in range(img_num):
            pred=self.defend_single(imgs[i])
            if pred==labels[i]: correct+=1
        return correct

    def defend_preprocess(self,img):
        if isinstance(img,torch.Tensor): img=img.numpy()
        if img.ndim==3 and img.shape[-1]==img.shape[-2]:img=img.transpose(1,2,0)
        img=np.uint8(img*255)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.bilateralFilter(img,3,self.std1,self.std2)
        img=cv2.bilateralFilter(img,5,self.std1,self.std2)
        return img

    def defend_inprocess(self,img):
        rb=np.random.random()*self.base-self.base/2
        rc=np.random.random()*20-10
        img_tmp=cv2.bilateralFilter(img,self.base-1,1,self.base*i+rb)
        img_tmp=Image.fromarray(cv2.cvtColor(img_tmp,cv2.COLOR_BGR2RGB))
        img_tmp=img_tmp.rotate(rc,expand=1.1)
        return img

def save_img(x,img_name):
    if isinstance(x,torch.Tensor):
        if 'cpu'!=str(x.device): x=x.detach().cpu()
        x=x.numpy()
    if x.ndim==4: x=x[0]
    if x.shape[-1]==x.shape[-2]: x=x.transpose(1,2,0)
    x=np.clip(np.round(x*255),0,255)
    x_pil=Image.fromarray(np.uint8(x))
    x_pil.save(img_name)

def forward_proprecess(x,mean,std):
    # norm = transforms.Normalize(mean,std)
    # x=norm(x.unsqueeze(0))
    # x=(x-mean.reshape(-1, 1, 1))/std.reshape(-1, 1, 1)
    # x=x.unsqueeze(0)
    if isinstance(x,np.ndarray): x=torch.from_numpy(x)
    if len(x.shape)==3: x=x.unsqueeze(0)
    if x.shape[-2]==x.shape[-3]: x=x.permute(0,3,1,2)
    if x.max()>10: x=x/255.0
    x=(x-mean.reshape(1,-1, 1, 1))/std.reshape(1,-1, 1, 1)
    x=x.requires_grad_(True).cuda()
    return x

def inverse_proprecess(x,mean,std):
    x=x.squeeze(0).detach().cpu()#.permute(1,2,0).numpy()
    x=x*std.reshape(-1, 1, 1)+mean.reshape(-1, 1, 1)
    return x

def loss_grad_bpda(model,x_orig,x_adv_bpda,x_adv_def,y_adv,mean,std):
    mean=torch.from_numpy(mean).cuda()
    std=torch.from_numpy(std).cuda()
    logits=model(x_adv_def.cuda())
    logits=softmax(logits,dim=1)
    # _, preds = torch.max(logits.data,1)
    grads=[]
    def save_grad():
        def hook(grad):
            # grads.append(grad.squeeze().permute(1,2,0).cpu().numpy().copy())
            grads.append(grad.cpu().numpy().copy())
            grad.data.zero_()
        return hook
    handle=x_adv_def.register_hook(save_grad())

    diff=(x_adv_bpda.cuda()-x_orig.cuda())*std[None,:, None, None]
    # x_adv_def_flip=torch.flip(x_adv_def,[1])
    # diff = x_adv_def_flip.cuda()-x_orig.cuda()
    normalized_L2=torch.sqrt(torch.mean(diff*diff))
    ce=CrossEntropyLoss()
    xent=ce(logits,one_hot(torch.LongTensor([y_adv]),logits.shape[1]).float().cuda())
    loss=xent+LAM*max(normalized_L2 - EPSILON, 0.0)
    loss.backward()
    model.zero_grad()

    handle.remove()
    grad=torch.from_numpy(grads[0]).cuda()
    # grad=grad/(torch.sum(torch.abs(grad),dim=tuple(range(1,len(x_adv_bpda.shape))),keepdim=True)+1e-8)
    return logits.detach().cpu().numpy(),grad,normalized_L2.detach().cpu().numpy()

# def loss_grad_bpdaeot(model,x_orig,x_adv,y_adv,mean,std,defend=None):
#     mean=torch.from_numpy(mean).cuda()
#     std=torch.from_numpy(std).cuda()
#     if defend:  
#             x_adv_def=[]
#             for _ in range(ENSEMBLE_SIZE):
#                 x_adv_def_tmp=tctensorGD_warp(x_adv*std[None,:,None,None]+mean[None,:,None,None])
#                 x_adv_def.append(x_adv_def_tmp)
#             x_adv_def=torch.vstack(x_adv_def)
#             x_adv_def=(x_adv_def-mean[None,:,None,None])/std[None,:,None,None]

#             x_adv_def1=x_adv*std[None,:,None,None]+mean[None,:,None,None]
#             x_adv_def1,_=defend(x_adv_def1.detach().cpu())
#             x_adv_def1=(torch.from_numpy(x_adv_def1).permute(0,3,1,2).cuda()-mean[None,:,None,None])/std[None,:,None,None]
#             # x_adv_def = torch.stack([defend(x_adv_bpdaeot).permute(2,0,1).squeeze(0) for _ in range(ENSEMBLE_SIZE)], axis=0)
#     else:
#         x_adv_def=x_adv.unsqueeze(0).repeat(ENSEMBLE_SIZE,1)

#     logits1=model(x_adv_def1.cuda())
#     # _, preds1 = torch.max(logits1.data,1)
#     logits=model(x_adv_def.cuda())
#     grads=[]
#     def save_grad():
#         def hook(grad):
#             # grads.append(grad.squeeze().permute(1,2,0).cpu().numpy().copy())
#             grads.append(grad.cpu().numpy().copy())
#             grad.data.zero_()
#         return hook
#     handle=x_adv.register_hook(save_grad())

#     diff=(x_adv.cuda()-x_orig.cuda())*std[None,:, None, None]
#     # x_adv_def_flip=torch.flip(x_adv_def,[1])
#     # diff = x_adv_def_flip.cuda()-x_orig.cuda()
#     normalized_L2=torch.sqrt(torch.mean(diff*diff))
#     ce=CrossEntropyLoss()
#     xent=ce(logits,one_hot(torch.LongTensor([y_adv]),logits.shape[1]).repeat(logits.shape[0],1).float().cuda())
#     loss=xent+LAM*max(normalized_L2 - EPSILON, 0.0)
#     loss.backward()
#     model.zero_grad()

#     handle.remove()
#     grad=torch.from_numpy(grads[0]).cuda()
#     return logits1.data,grad,normalized_L2.detach().cpu().numpy()

# def loss_grad_eot(model,x_orig,x_adv,y_adv,defend,mean,std):
#     mean=torch.from_numpy(mean).cuda()
#     std=torch.from_numpy(std).cuda()
#     ensemble_xs = torch.stack([defend(x_adv.squeeze(0).permute(1,2,0)*std+mean).permute(2,0,1).squeeze(0) for _ in range(ENSEMBLE_SIZE)], axis=0)
#     ensemble_xs = (ensemble_xs-mean[None,:, None, None])/std[None,:, None, None]
#     ensemble_logits=model(ensemble_xs.cuda())
#     _, ensemble_preds = torch.max(ensemble_logits.data,1)
#     grads=[]
#     def save_grad():
#         def hook(grad):
#             # grads.append(grad.squeeze().permute(1,2,0).cpu().numpy().copy())
#             grads.append(grad.cpu().numpy().copy())
#             grad.data.zero_()
#         return hook
#     handle=x_adv.register_hook(save_grad())

#     diff=x_adv.cuda()-x_orig.cuda()
#     normalized_L2=torch.sqrt(torch.mean(diff*diff))
#     ce=CrossEntropyLoss()
#     xent=ce(ensemble_logits,torch.tile(one_hot(torch.LongTensor([y_adv]),ensemble_logits.shape[1]).float().cuda(), (ensemble_logits.shape[0], 1)))
#     loss=xent+LAM*max(normalized_L2 - EPSILON, 0.0)
#     loss.backward()
#     model.zero_grad()

#     handle.remove()
#     grad=torch.from_numpy(grads[0]).cuda()
#     return grad,normalized_L2.detach().cpu().numpy()

def Non_attack(model,x_orig,y_orig,mean,std):
    x_orig_tc=forward_proprecess(x_orig,mean,std)
    mean=torch.from_numpy(mean).cuda()
    std=torch.from_numpy(std).cuda()
    logits=model(x_orig_tc.cuda())
    _, preds = torch.max(logits.data,1)
    if preds!=y_orig:
        ret=1
    return ret

def BPDA(model,x_orig,y_orig,mean,std,classes=10,epoch=3,defend_pre=None,defend_in=None,saved_dir=None):
    acc=[]
    att=[]
    L2s_all=[]
    # x_adv = x_orig.clone()
    x_adv_bpda = x_orig.clone()
    y_adv = list(range(classes))
    y_adv.remove(y_orig.cpu())
    y_adv = np.random.choice(y_adv)
    if saved_dir: save_img(x_orig,saved_dir+'/orig.png')
    for i in range(epoch):
        x_adv_def=defend_pre(x_adv_bpda)
        x_orig_tc=forward_proprecess(x_orig,mean,std)
        x_adv_bpda_tc=forward_proprecess(x_adv_bpda,mean,std)
        logits=[]
        gs=[]
        L2s=[]
        for j in range(10):
            x_adv_def_tmp=defend_in(x_adv_def.copy())
            x_adv_def_tc=forward_proprecess(x_adv_def_tmp,mean,std)
            logit_tmp,g_tmp,L2_tmp=loss_grad_bpda(model,x_orig_tc,x_adv_bpda_tc,x_adv_def_tc,y_adv,mean,std)
            logits.append(logit_tmp)
            gs.append(g_tmp)
            L2s.append(L2_tmp)
        logits=np.vstack(logits)
        gs=torch.vstack(gs)
        L2s=np.vstack(L2s)

        p=np.argmax(logits.mean(axis=0))
        g=gs.mean(axis=0)
        L2=L2s.mean(axis=0)
        x_adv_bpda_tc -= LR * g
        x_adv_bpda = inverse_proprecess(x_adv_bpda_tc,mean,std)
        x_adv_bpda = torch.clip(x_adv_bpda, 0, 1)
        print('step %d, gt=%d, pred=%d, L2:%.3f' % (i, y_orig, p, L2))
        if p!=y_orig:# and L2<EPSILON:
            acc.append(0)
        else:
            acc.append(1)
        if p==y_adv:
            att.append(1)
        else:
            att.append(0)
        L2s_all.append(L2)
    return np.array(acc).reshape(1,-1),np.array(att).reshape(1,-1),np.array(L2s_all).reshape(1,-1)

# def BPDA_EOT(model,x_orig,y_orig,mean,std,classes=10,epoch=3,defend=None,saved_dir=None):
#     acc=[]
#     att=[]
#     L2s=[]
#     # x_adv = x_orig.clone()
#     x_adv_bpdaeot = x_orig.clone()
#     y_adv = list(range(classes))
#     y_adv.remove(y_orig.cpu())
#     y_adv = np.random.choice(y_adv)
#     if saved_dir: save_img(x_orig,saved_dir+'/orig.png')
#     for i in range(epoch):
#         if saved_dir: save_img(x_adv_bpdaeot,saved_dir+'/bpdaeot_'+str(i)+'.png')
#         # if defend:  
#         #     x_adv_def=[]
#         #     for _ in range(ENSEMBLE_SIZE):
#         #         x_adv_def_tmp,_=defend(x_adv_bpdaeot)
#         #         if x_adv_def_tmp.ndim==3: x_adv_def_tmp=np.expand_dims(x_adv_def_tmp,axis=0)
#         #         x_adv_def.append(x_adv_def_tmp.transpose(0,3,1,2))
#         #     x_adv_def=np.vstack(x_adv_def)
#         #     # x_adv_def = torch.stack([defend(x_adv_bpdaeot).permute(2,0,1).squeeze(0) for _ in range(ENSEMBLE_SIZE)], axis=0)
#         # else:
#         #     x_adv_def=x_adv_bpdaeot
#         # if saved_dir: save_img(x_adv_def,saved_dir+'/def_'+str(i)+'.png')
#         x_orig_tc=forward_proprecess(x_orig,mean,std)
#         # x_adv_def_tc=forward_proprecess(x_adv_def,mean,std)
#         x_adv_bpdaeot_tc=forward_proprecess(x_adv_bpdaeot,mean,std)

#         p,g,L2=loss_grad_bpdaeot(model,x_orig_tc,x_adv_bpdaeot_tc,y_adv,mean,std,defend)
#         x_adv_bpdaeot_tc -= LR * g
#         x_adv_bpdaeot = inverse_proprecess(x_adv_bpdaeot_tc,mean,std)
#         x_adv_bpdaeot = torch.clip(x_adv_bpdaeot, 0, 1)
#         logger.info('step %d, gt=%d, pred=%d, L2:%.3f' % (i, y_orig, p, L2))
#         if p!=y_orig:# and L2<EPSILON:
#             acc.append(0)
#         else:
#             acc.append(1)
#         if p==y_adv:
#             att.append(1)
#         else:
#             att.append(0)
#         L2s.append(L2)
#     return np.array(acc).reshape(1,-1),np.array(att).reshape(1,-1),np.array(L2s).reshape(1,-1)

# def BPDA_EOT(model,x_orig,y_orig,mean,std,classes=10,epoch=3,defend=None):
#     ret=0
#     x_adv = x_orig.clone()
#     y_adv = list(range(classes))
#     y_adv.remove(y_orig.cpu())
#     y_adv = np.random.choice(y_adv)
#     for i in range(epoch):
#         # if defend:  x_adv=defend(x_adv)
#         x_orig_tc=forward_proprecess(x_orig,mean,std)
#         x_adv_tc=forward_proprecess(x_adv,mean,std)
#         logits=model(x_adv_tc.cuda())
#         _, p = torch.max(logits.data,1)

#         g,L2=loss_grad_eot(model,x_orig_tc,x_adv_tc,y_adv,defend,mean,std)
#         x_adv_tc -= LR * g
#         x_adv = inverse_proprecess(x_adv_tc,mean,std)
#         x_adv = torch.clip(x_adv, 0, 1)
#         print('step %d, gt=%d, pred=%d, L2:%.3f' % (i, y_orig, p, L2))
#         if p!=y_orig and L2<EPSILON:
#             ret=1
#             break
    
#     return ret

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_data', type=str, default='vgg16_imagenet', help='image name')
    parser.add_argument('--device', type=str, default='0', help='image name')
    parser.add_argument('--data_num', type=float, default=0.0001, help='image name')
    parser.add_argument('--attacker', default='bpda', choices=['bpda', 'bpda_eot','none'])
    parser.add_argument('--defender', default='MFIR', choices=['MFIR'])
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--ensemble_size', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--saved_img_num', type=int, default=10)
    args = parser.parse_args()

    LR=args.lr
    LAM=args.lam
    EPSILON = args.epsilon
    ENSEMBLE_SIZE=args.ensemble_size

    model_vanilla_type=args.model_data
    os.environ['CUDA_VISIBLE_DEVICES']=args.device
    saved_dir=os.path.join('../saved_tests/BPDA',args.model_data,args.attacker,str(args.epsilon),str(args.epoch),args.defender)
    # saved_dir = '../saved_tests/BPDA/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    logger=logging.getLogger(name='r')
    logger.setLevel(logging.FATAL)
    formatter=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    fh=logging.FileHandler(os.path.join(saved_dir,'log_'+args.attacker+'_acc.txt'))
    fh.setLevel(logging.FATAL)
    fh.setFormatter(formatter)
    
    ch=logging.StreamHandler()
    ch.setLevel(logging.FATAL)
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    # logger.fatal(('\n----------defense record-----------'))

    '''
    加载cifar-10图像
    '''
    g.setup_seed(0)
    if 'imagenet' in model_vanilla_type:
        dataset_name='imagenet'
    else:
        dataset_name='cifar-10'
    data_setting=g.dataset_setting(dataset_name)
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,'val',10)#data_setting.hyperopt_img_val_num)
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=data_setting.workers, pin_memory=True)    

    '''
    加载模型
    '''
    dir_model  = '../models/cifar_vanilla_'+model_vanilla_type+'.pth.tar'
    model,_=g.select_model(model_vanilla_type, dir_model)
    model.eval()

    '''
    防御初始化
    '''
    defender=MFIR_defender(model,data_setting.mean, data_setting.std,data_setting.input_shape)


    '''
    加载图像
    '''
    mean=data_setting.mean
    std=data_setting.std
    classes=data_setting.nb_classes
    # defend=defences[args.defender]

    num_correct=0
    accs=[]
    atts=[]
    L2s=[]
    for i,(x_orig,y_orig) in enumerate(tqdm(dataloader)):
        #  break
        x_orig=x_orig[0]
        y_orig=y_orig[0]
        
        saved_dir_tmp=None
        if i<args.saved_img_num:
            saved_dir_tmp=os.path.join(saved_dir,str(i))
            if os.path.exists(saved_dir_tmp):
                shutil.rmtree(saved_dir_tmp)
            os.makedirs(saved_dir_tmp)
        if args.attacker=='bpda':
            acc,att,L2=BPDA(model,x_orig,y_orig,mean,std,classes=classes,epoch=args.epoch,defend_pre=defender.defend_preprocess,defend_in=defender.defend_inprocess,saved_dir=saved_dir_tmp)#defend_FD_ago_warp)#
        elif args.attacker=='bpda_eot':
            acc,att,L2=BPDA_EOT(model,x_orig,y_orig,mean,std,classes=classes,epoch=args.epoch,defend=defend,saved_dir=saved_dir_tmp)#defend_FD_ago_warp)#
        elif args.attacker=='none':
            ret=Non_attack(model,x_orig,y_orig,mean,std)
        else:
            logging.raiseExceptions('Wrong attacker')
        accs.append(acc)
        atts.append(att)
        L2s.append(L2)
        print('\n\n{}'.format(i))
    accs_np=np.vstack(accs)
    atts_np=np.vstack(atts)
    L2s_np=np.vstack(L2s)
    
    acc_r=accs_np.sum(axis=0)*100/len(dataset)
    att_r=atts_np.sum(axis=0)*100/len(dataset)
    L2_r=L2s_np.mean(axis=0)
    prt='acc: {}'.format(acc_r)
    # print(prt)
    logger.fatal(prt)

    prt='att: {}'.format(att_r)
    # print(prt)
    logger.fatal(prt)

    prt='L2: {}'.format(L2_r)
    # print(prt)
    logger.fatal(prt)



