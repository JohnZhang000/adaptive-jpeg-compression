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

from defense import defend_webpf_wrap,defend_webpf_my_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
from defense import defend_my_webpf,defend_jpeg_wrap
from defense_ago import defend_FD_ago_warp,defend_my_fd_ago
from fd_jpeg import fddnn_defend
from adaptivce_defense import adaptive_defender

from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin

import argparse



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
    x=(x-mean.reshape(-1, 1, 1))/std.reshape(-1, 1, 1)
    x=x.unsqueeze(0)
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
    _, preds = torch.max(logits.data,1)
    grads=[]
    def save_grad():
        def hook(grad):
            # grads.append(grad.squeeze().permute(1,2,0).cpu().numpy().copy())
            grads.append(grad.cpu().numpy().copy())
            grad.data.zero_()
        return hook
    handle=x_adv_def.register_hook(save_grad())

    diff=(x_adv_def.cuda()-x_orig.cuda())*std[None,:, None, None]
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
    return preds,grad,normalized_L2.detach().cpu().numpy()

def loss_grad_eot(model,x_orig,x_adv,y_adv,defend,mean,std):
    mean=torch.from_numpy(mean).cuda()
    std=torch.from_numpy(std).cuda()
    ensemble_xs = torch.stack([defend(x_adv.squeeze(0).permute(1,2,0)*std+mean).permute(2,0,1).squeeze(0) for _ in range(ENSEMBLE_SIZE)], axis=0)
    ensemble_xs = (ensemble_xs-mean[None,:, None, None])/std[None,:, None, None]
    ensemble_logits=model(ensemble_xs.cuda())
    _, ensemble_preds = torch.max(ensemble_logits.data,1)
    grads=[]
    def save_grad():
        def hook(grad):
            # grads.append(grad.squeeze().permute(1,2,0).cpu().numpy().copy())
            grads.append(grad.cpu().numpy().copy())
            grad.data.zero_()
        return hook
    handle=x_adv.register_hook(save_grad())

    diff=x_adv.cuda()-x_orig.cuda()
    normalized_L2=torch.sqrt(torch.mean(diff*diff))
    ce=CrossEntropyLoss()
    xent=ce(ensemble_logits,torch.tile(one_hot(torch.LongTensor([y_adv]),ensemble_logits.shape[1]).float().cuda(), (ensemble_logits.shape[0], 1)))
    loss=xent+LAM*max(normalized_L2 - EPSILON, 0.0)
    loss.backward()
    model.zero_grad()

    handle.remove()
    grad=torch.from_numpy(grads[0]).cuda()
    return grad,normalized_L2.detach().cpu().numpy()

def Non_attack(model,x_orig,y_orig,mean,std):
    x_orig_tc=forward_proprecess(x_orig,mean,std)
    mean=torch.from_numpy(mean).cuda()
    std=torch.from_numpy(std).cuda()
    logits=model(x_orig_tc.cuda())
    _, preds = torch.max(logits.data,1)
    if preds!=y_orig:
        ret=1
    return ret

def BPDA(model,x_orig,y_orig,mean,std,classes=10,epoch=3,defend=None,saved_dir=None):
    ret=0
    # x_adv = x_orig.clone()
    x_adv_bpda = x_orig.clone()
    y_adv = list(range(classes))
    y_adv.remove(y_orig.cpu())
    y_adv = np.random.choice(y_adv)
    if saved_dir: save_img(x_orig,saved_dir+'/orig.png')
    for i in range(epoch):
        if saved_dir: save_img(x_adv_bpda,saved_dir+'/bpda_'+str(i)+'.png')
        if defend:  
            x_adv_def,_=defend(x_adv_bpda)
            x_adv_def=torch.from_numpy(x_adv_def).squeeze(0).permute(2,0,1)
        else:
            x_adv_def=x_adv_bpda
        if saved_dir: save_img(x_adv_def,saved_dir+'/def_'+str(i)+'.png')
        x_orig_tc=forward_proprecess(x_orig,mean,std)
        x_adv_def_tc=forward_proprecess(x_adv_def,mean,std)
        x_adv_bpda_tc=forward_proprecess(x_adv_bpda,mean,std)

        p,g,L2=loss_grad_bpda(model,x_orig_tc,x_adv_bpda_tc,x_adv_def_tc,y_adv,mean,std)
        x_adv_def_tc -= LR * g
        x_adv_bpda = inverse_proprecess(x_adv_def_tc,mean,std)
        x_adv_bpda = torch.clip(x_adv_bpda, 0, 1)
        # print('step %d, gt=%d, pred=%d, L2:%.3f' % (i, y_orig, p, L2))
        if p!=y_orig and L2<EPSILON:
            ret=1
            break
    return ret

def BPDA_EOT(model,x_orig,y_orig,mean,std,classes=10,epoch=3,defend=None):
    ret=0
    x_adv = x_orig.clone()
    y_adv = list(range(classes))
    y_adv.remove(y_orig.cpu())
    y_adv = np.random.choice(y_adv)
    for i in range(epoch):
        # if defend:  x_adv=defend(x_adv)
        x_orig_tc=forward_proprecess(x_orig,mean,std)
        x_adv_tc=forward_proprecess(x_adv,mean,std)
        logits=model(x_adv_tc.cuda())
        _, p = torch.max(logits.data,1)

        g,L2=loss_grad_eot(model,x_orig_tc,x_adv_tc,y_adv,defend,mean,std)
        x_adv_tc -= LR * g
        x_adv = inverse_proprecess(x_adv_tc,mean,std)
        x_adv = torch.clip(x_adv, 0, 1)
        print('step %d, gt=%d, pred=%d, L2:%.3f' % (i, y_orig, p, L2))
        if p!=y_orig and L2<EPSILON:
            ret=1
            break
    return ret

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_data', type=str, default='resnet50_imagenet', help='image name')
    parser.add_argument('--device', type=str, default='3', help='image name')
    parser.add_argument('--data_num', type=float, default=0.002, help='image name')
    parser.add_argument('--attacker', default='bpda', choices=['bpda', 'bpda_eot','none'])
    parser.add_argument('--defender', default='AGO', choices=['GauA','BDR','RDG','WEBPF','JPEG','SHIELD','FD','FDD','AGO','Ours'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--ensemble_size', type=int, default=30)
    parser.add_argument('--saved_img_num', type=int, default=5)
    args = parser.parse_args()

    LR=args.lr
    LAM=args.lam
    EPSILON = args.epsilon
    ENSEMBLE_SIZE=args.ensemble_size

    model_vanilla_type=args.model_data
    os.environ['CUDA_VISIBLE_DEVICES']=args.device
    saved_dir = '../saved_tests/img_attack/'+model_vanilla_type
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
    
    logger.fatal(('\n----------defense record-----------'))

    '''
    加载cifar-10图像
    '''
    g.setup_seed(0)
    if 'imagenet' in model_vanilla_type:
        dataset_name='imagenet'
    else:
        dataset_name='cifar-10'
    data_setting=g.dataset_setting(dataset_name)
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,'val',args.data_num)
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
    defences={}
    defences['GauA']=GaussianAugmentation(sigma=0.01,augmentation=False)
    defences['BDR']=SpatialSmoothing()
    defences['RDG']=defend_rdg_wrap
    defences['WEBPF']=defend_webpf_wrap
    defences['JPEG']=defend_jpeg_wrap
    defences['SHIELD']=defend_shield_wrap
    defences['FD']=defend_fd_wrap
    defences['FDD']=fddnn_defend
    defences['AGO']=defend_FD_ago_warp
    
    table_pkl=os.path.join(saved_dir,'table_dict.pkl')
    gc_model_dir=os.path.join(saved_dir,'model_best.pth.tar')
    model_mean_std=os.path.join(saved_dir,'mean_std_train.npy')
    adaptive_defender=adaptive_defender(table_pkl,gc_model_dir,data_setting.nb_classes,data_setting.input_shape[-1],data_setting.pred_batch_size,model_mean_std)
    defences['Ours']=adaptive_defender.defend

    '''
    加载图像
    '''
    mean=data_setting.mean
    std=data_setting.std
    classes=data_setting.nb_classes
    defend=defences[args.defender]

    num_correct=0
    for i,(x_orig,y_orig) in enumerate(tqdm(dataloader)):
        #  break
        x_orig=x_orig[0]
        y_orig=y_orig[0]
        
        saved_dir=None
        if i<args.saved_img_num:
            saved_dir=os.path.join(saved_dir,'imgs',args.attacker,args.defender,str(i))
            if os.path.exists(saved_dir):
                shutil.rmtree(saved_dir)
            os.makedirs(saved_dir)
        if args.attacker=='bpda':
            ret=BPDA(model,x_orig,y_orig,mean,std,classes=classes,epoch=50,defend=defend,saved_dir=saved_dir)#defend_FD_ago_warp)#
        elif args.attacker=='bpda_eot':
            ret=BPDA_EOT(model,x_orig,y_orig,mean,std,classes=classes,epoch=50,defend=tctensorGD)
        elif args.attacker=='none':
            ret=Non_attack(model,x_orig,y_orig,mean,std)

        else:
            logging.raiseExceptions('Wrong attacker')

        num_correct+=(1-ret)
    logger.info('Attacker:{} Defender:{} Acc:{}/{} {}'.format(args.attacker,args.defender,num_correct,len(dataset),num_correct/len(dataset)))



