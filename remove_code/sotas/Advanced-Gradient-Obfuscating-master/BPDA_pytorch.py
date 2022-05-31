import os
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
from defense import defend_my_webpf
from defense_ago import defend_FD_ago_warp,defend_my_fd_ago
from fd_jpeg import fddnn_defend
from adaptivce_defense import adaptive_defender

from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin



LR=0.1
LAM=1
EPSILON = 0.05
ENSEMBLE_SIZE=30

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

def loss_grad_bpda(model,x_orig,x_adv_bpda,x_adv_def,y_adv):
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

    diff=x_adv_def.cuda()-x_orig.cuda()
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

def BPDA(model,x_orig,y_orig,mean,std,classes=10,epoch=3,defend=None):
    ret=0
    # x_adv = x_orig.clone()
    x_adv_bpda = x_orig.clone()
    y_adv = list(range(classes))
    y_adv.remove(y_orig.cpu())
    y_adv = np.random.choice(y_adv)
    for i in range(epoch):
        if defend:  
            x_adv_def,_=defend(x_adv_bpda)
            x_adv_def=torch.from_numpy(x_adv_def).squeeze(0).permute(2,0,1)
        else:
            x_adv_def=x_adv_bpda
        x_orig_tc=forward_proprecess(x_orig,mean,std)
        x_adv_def_tc=forward_proprecess(x_adv_def,mean,std)
        x_adv_bpda_tc=forward_proprecess(x_adv_bpda,mean,std)

        p,g,L2=loss_grad_bpda(model,x_orig_tc,x_adv_bpda_tc,x_adv_def_tc,y_adv)
        x_adv_def_tc -= LR * g
        x_adv_bpda = inverse_proprecess(x_adv_def_tc,mean,std)
        x_adv_bpda = torch.clip(x_adv_bpda, 0, 1)
        print('step %d, gt=%d, pred=%d, L2:%.3f' % (i, y_orig, p, L2))
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
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    # 配置解释器参数
    if len(sys.argv)!=2:
        print('Manual Mode !!!')
        model_vanilla_type    = 'resnet50_imagenet'
    else:
        print('Terminal Mode !!!')
        model_vanilla_type    = sys.argv[1]
    
    saved_dir = '../saved_tests/img_attack/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    logger=logging.getLogger(name='r')
    logger.setLevel(logging.FATAL)
    formatter=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s -%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    fh=logging.FileHandler(os.path.join(saved_dir,'log_bpda_acc.txt'))
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
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,'val',0.0002)
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

    defences_pre=[]
    defences_names_pre=[]
    # defences_pre.append(GaussianAugmentation(sigma=0.01,augmentation=False))
    # defences_names_pre.append('GauA')
    # defences_pre.append(SpatialSmoothing())
    # defences_names_pre.append('BDR')
    # defences_pre.append(defend_rdg_wrap)
    # defences_names_pre.append('RDG')
    # defences_pre.append(defend_webpf_wrap)
    # defences_names_pre.append('WEBPF')
    # defences_pre.append(JpegCompression(clip_values=(0,1),quality=25,channels_first=False))
    # defences_names_pre.append('JPEG')
    # defences_pre.append(defend_shield_wrap)
    # defences_names_pre.append('SHIELD')
    # defences_pre.append(defend_fd_wrap)
    # defences_names_pre.append('EPA')
    # defences_pre.append(fddnn_defend)
    # defences_names_pre.append('FD')
    # defences_pre.append(defend_FD_ago_warp)
    # defences_names_pre.append('FD_ago')
    
    table_pkl=os.path.join(saved_dir,'table_dict.pkl')
    gc_model_dir=os.path.join(saved_dir,'model_best.pth.tar')
    model_mean_std=os.path.join(saved_dir,'mean_std_train.npy')
    # threshs=[0.001,0.001,0.001]
    # fd_ago_new=defend_my_fd_ago(table_pkl,gc_model_dir,[0.3,0.8,0.8],[0.0001,0.0001,0.0001],model_mean_std)
    # fd_ago_new.get_cln_dct(images.transpose(0,2,3,1).copy())
    # print(fd_ago_new.abs_threshs)
    # defences_pre.append(fd_ago_new.defend)
    # defences_names_pre.append('fd_ago_my')
    # defences_pre.append(fd_ago_new.defend_channel_wise_with_eps)
    # defences_names_pre.append('fd_ago_my')
    # defences_pre.append(fd_ago_new.defend_channel_wise)
    # defences_names_pre.append('fd_ago_my_no_eps')
    # defences_pre.append(fd_ago_new.defend_channel_wise_adaptive_table)
    # defences_names_pre.append('fd_ago_my_ada')
    adaptive_defender=adaptive_defender(table_pkl,gc_model_dir,data_setting.nb_classes,data_setting.input_shape[-1],data_setting.pred_batch_size,model_mean_std)
    
    # defences_pre.append(adaptive_defender.defend)
    # defences_names_pre.append('ADAD')
    # defences_pre.append(adaptive_defender.defend)
    # defences_names_pre.append('ADAD-flip')
    # defences_pre.append(adaptive_defender.defend)
    # defences_names_pre.append('ADAD+eps-flip')
    defences_pre.append(adaptive_defender.defend)
    defences_names_pre.append('ADAD+eps+flip')

    '''
    加载图像
    '''
    # x_orig=Image.open('./sotas/Advanced-Gradient-Obfuscating-master/cat.jpg').resize((224,224))
    mean=data_setting.mean
    std=data_setting.std
    classes=data_setting.nb_classes
    # x_orig=np.array(x_orig,dtype=np.float32)/255.0
    # defend=adaptive_defender.defend
    defend=defend_FD_ago_warp

    num_success=0
    for i,(x_orig,y_orig) in enumerate(tqdm(dataloader)):
        # if i>10: break
        x_orig=x_orig[0]
        y_orig=y_orig[0]
        
        ret=BPDA(model,x_orig,y_orig,mean,std,classes=classes,epoch=50,defend=defend)#defend_FD_ago_warp)#
        # ret=BPDA_EOT(model,x_orig,y_orig,mean,std,classes=classes,epoch=50,defend=tctensorGD)

        num_success+=ret
        # print('sample:{},succ:{}'.format(i,ret))
    print('success_rate:{}/{} {}'.format(num_success,len(dataset),num_success/len(dataset)))





