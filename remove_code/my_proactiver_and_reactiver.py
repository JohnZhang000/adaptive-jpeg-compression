#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:06:22 2021

@author: ubuntu204
"""
import sys
sys.path.append("..")

import torch 
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
from models.cifar.allconv import AllConvNet
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from art.attacks.evasion import FastGradientMethod,DeepFool
from art.attacks.evasion import CarliniL2Method,CarliniLInfMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from my_classifier import Net #as spectrum_net_cifar
# from my_classifier import Net_imagenet as spectrum_net_imagenet
from my_spectrum_analyzer import img_spectrum_analyzer
from train_code.my_img_transformer import img_transformer
from tqdm import tqdm
from art.defences.preprocessor import GaussianAugmentation, JpegCompression,FeatureSqueezing,LabelSmoothing,Resample,SpatialSmoothing,ThermometerEncoding,TotalVarMin
from defense import defend_webpf_wrap,defend_rdg_wrap,defend_fd_wrap,defend_bdr_wrap,defend_shield_wrap
from defense_ago import defend_FD_ago_warp
from load_cifar_data import load_CIFAR_batch,load_CIFAR_train,load_imagenet_batch,load_imagenet_filenames


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import joblib
import logging
sys.path.append('../common_code')
import general as g

sys.path.append('../common_code')
from load_cifar_data import load_CIFAR_batch#,load_CIFAR_train

def get_accuracy(pred, label_gt, label_adv, adv_cls):
    pred      = pred#.detach().cpu().numpy()
    label_gt  = label_gt#.detach().cpu().numpy()
    label_adv = label_adv#.detach().cpu().numpy()
    idx_correct_pred = (pred == label_gt)
    idx_correct_adv  = (pred == label_adv*adv_cls)
    correct_pred =idx_correct_pred.sum().item()
    correct_adv  = idx_correct_adv.sum().item()
    correct = correct_pred+correct_adv
    return correct,correct_adv,correct_pred

# def select_attack(fmodel, att_method, eps):
#     if att_method   == 'FGSM_L2_IDP':
#         attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=2)
#     elif att_method == 'PGD_L2_IDP':
#         attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=2,batch_size=512,verbose=False)
#     elif att_method == 'CW_L2_IDP':
#         attack = CarliniL2Method(classifier=fmodel,batch_size=512,verbose=False)
#     elif att_method == 'Deepfool_L2_IDP':
#         attack = DeepFool(classifier=fmodel,batch_size=512,verbose=False)
        
#     elif att_method == 'FGSM_Linf_IDP':
#         attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=np.inf)
#     elif att_method == 'PGD_Linf_IDP':
#         attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=np.inf,batch_size=512,verbose=False)
#     elif att_method == 'CW_Linf_IDP':
#         attack = CarliniLInfMethod(classifier=fmodel,eps=eps,batch_size=512,verbose=False)
    
#     elif att_method == 'FGSM_L2_UAP':
#         attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
#     elif att_method == 'PGD_L2_UAP':
#         attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
#     elif att_method == 'CW_L2_UAP':
#         attack = UniversalPerturbation(classifier=fmodel,attacker='carlini',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
#     elif att_method == 'Deepfool_L2_UAP':
#         attack = UniversalPerturbation(classifier=fmodel,attacker='deepfool',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
    
#     elif att_method == 'FGSM_Linf_UAP':
#         attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=512,verbose=False)
#     elif att_method == 'PGD_Linf_UAP':
#         attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=512,verbose=False)
#     elif att_method == 'CW_Linf_UAP':
#         attack = UniversalPerturbation(classifier=fmodel,attacker='carlini_inf',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=512,verbose=False)
    
#     else:
#         raise Exception('Wrong Attack Mode: {} !!!'.format(att_method))
#     return attack, eps
    
def get_attacked_img(fmodel, images, labels, batch_size, attack_idx, eps_idx,dataset):
    attack,eps=g.select_attack(fmodel, attack_idx,eps_idx)
    
    batch_num       = int(len(labels)/batch_size) 
    images_adv_list=[]
    for i in tqdm(range(batch_num)):
        start_idx=batch_size*i
        end_idx=min(batch_size*(i+1),len(labels))
        if 'cifar-10'==dataset:
            images_batch = images[start_idx:end_idx,...].transpose(0,3,1,2)
            labels_batch = labels[start_idx:end_idx,...]
        elif 'imagenet'==dataset:
            images_batch,labels_batch=load_imagenet_batch(i,batch_size,data_dir,images,labels)
        images_adv = attack.generate(x=images_batch,y=labels_batch)
        images_adv_list.append(images_adv)
    images_advs=np.vstack(images_adv_list)
    return images_advs       
 
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     # random.seed(seed)
#     torch.backends.cudnn.deterministic = True
    
class proactiver_and_reactiver(nn.Module):
    def __init__(self,spectrum_ckpt, adaboost_pkl, svm_pkl, classifier_type, dataset):
        super(proactiver_and_reactiver,self).__init__()
        self.dataset=dataset
        if 'imagenet'==dataset:
            self.input_shape=g.input_shape_imagenet
            self.img_size=g.input_shape_imagenet[2]
            mean_now=g.mean_imagenet
            std_now=g.std_imagenet
            self.num_classes=g.nb_classes_imagenet
            self.spectrum_size=g.spectrum_num_cifar#int(self.img_size/2)
            self.spectrum_mean_std=[0,1.0]
            self.adv_cls_label=g.nb_classes_imagenet+1
            self.fft_transformer1 = img_transformer(g.levels_all_imagenet,g.levels_start_imagenet,g.levels_end_imagenet,img_size=self.img_size)
            self.fft_transformer2 = img_transformer(g.levels_all_imagenet,g.levels_start_imagenet,g.levels_end_imagenet+1,img_size=self.img_size)
            
            self.spectrum_net = Net(self.spectrum_size)
            checkpoint = torch.load(spectrum_ckpt)
            self.spectrum_net.load_state_dict(checkpoint['state_dict'])
            self.spectrum_net.cuda().eval()
            
        elif 'cifar-10'==dataset:
            self.input_shape=g.input_shape_imagenet
            self.img_size=g.input_shape_cifar[2]
            mean_now=g.mean_cifar
            std_now=g.std_cifar
            self.num_classes=g.nb_classes_cifar
            self.spectrum_size=g.spectrum_num_imagenet#int(self.img_size/2)
            self.spectrum_mean_std=[0.0, 1.0]
            self.fft_transformer1 = img_transformer(g.levels_all_cifar,g.levels_start_cifar,g.levels_end_cifar,img_size=self.img_size)
            self.fft_transformer2 = img_transformer(g.levels_all_cifar,g.levels_start_cifar,g.levels_end_cifar+1,img_size=self.img_size)
            self.adv_cls_label=g.nb_classes_imagenet+1
            
            self.spectrum_net = Net(self.spectrum_size)
            checkpoint = torch.load(spectrum_ckpt)
            self.spectrum_net.load_state_dict(checkpoint['state_dict'])
            self.spectrum_net.cuda().eval()
        else:
            print('ERROR DATASET')
        
        self.s_analyzer=img_spectrum_analyzer(self.img_size,self.spectrum_size).batch_get_spectrum_feature


        
        if adaboost_pkl:
            self.adabooster=joblib.load(adaboost_pkl)
        if svm_pkl:
            self.svmer=joblib.load(svm_pkl)
        
        dir_model  = '../models/cifar_vanilla_'+classifier_type+'.pth.tar'
        self.classifier,dataset=g.select_model(classifier_type, dir_model)
        # if classifier_type == 'allconv':
        #     self.classifier = AllConvNet(10).eval()
        #     dir_model  = '../models/cifar_vanilla_allconv.pth.tar'
        # elif classifier_type == 'densenet':
        #     self.classifier = densenet(num_classes=10).eval()
        #     dir_model  = '../models/cifar_vanilla_densenet.pth.tar'
        # elif classifier_type == 'wrn':
        #     self.classifier = WideResNet(40, 10, 2, 0.0).eval()
        #     dir_model  = '../models/cifar_vanilla_wrn.pth.tar'
        # elif classifier_type == 'resnext':
        #     self.classifier = resnext29(num_classes=10).eval()
        #     dir_model  = '../models/cifar_vanilla_resnext.pth.tar' 
        # else:
        #     print('Error model name!!!')
            
        self.trans_classifier=transforms.Compose([transforms.Normalize(mean=mean_now, std=std_now)])     
        # self.classifier = torch.nn.DataParallel(self.classifier).cuda()
        # checkpoint = torch.load(dir_model)
        # self.classifier.load_state_dict(checkpoint['state_dict'])

        # mean   = np.array((0.5,0.5,0.5),dtype=np.float32)
        # std    = np.array((0.5,0.5,0.5),dtype=np.float32)
        self.fmodel = PyTorchClassifier(model = self.classifier,nb_classes=self.num_classes,clip_values=(0,1),
                                   input_shape=self.input_shape,loss = nn.CrossEntropyLoss(),
                                   preprocessing=(mean_now, std_now)) 
        
    def forward(self,x, method, spectrum_classifier, preprocesser):
        self.flag_lowpass=1
        self.flag_detect=1
        if 0==method:
            self.flag_lowpass=0
            self.flag_detect=0
        elif 1==method:
            self.flag_detect=0
        elif 3==method:
            self.flag_lowpass=0
        else:
            raise Exception('Wrong method')
            
        assert x.shape[-2]==x.shape[-1]
        cls_adv     = np.array([1])   #需要标为对抗样本的类别
        
        with torch.no_grad():
            # 功率谱分类  
            spectrum  = self.s_analyzer(x)
                        
            if 'cnn'==spectrum_classifier:
                spectrum    = torch.from_numpy(spectrum).cuda()
                spectrum    = (spectrum-self.spectrum_mean_std[0])/self.spectrum_mean_std[1]
                y_pred      = self.spectrum_net(spectrum)
                _,label_spectrum_before = torch.max(y_pred.data,dim=1)
                label_spectrum_before=label_spectrum_before.detach().cpu().numpy()
            elif 'adb'==spectrum_classifier:
                label_spectrum_before  = self.adabooster.predict(spectrum)
            elif 'svm'==spectrum_classifier:
                label_spectrum_before  = self.svmer.predict(spectrum)
            else:
                print('Wrong type classifier')
            
            # 图像变换   
            if self.flag_lowpass:
                if 'LPF1'==preprocesser:
                    x      = self.fft_transformer1.img_transform_tc(x)
                elif 'LPF2'==preprocesser:
                    x      = self.fft_transformer2.img_transform_tc(x)
                elif 'JPG'==preprocesser:
                    x,_=JpegCompression(clip_values=(0,1),quality=25,channels_first=False)(x.transpose(0,2,3,1))
                    x=x.transpose(0,3,1,2)
                elif 'GuA'==preprocesser:
                    x,_=GaussianAugmentation(sigma=0.01,augmentation=False)(x.transpose(0,2,3,1))
                    x=np.clip(x, 0, 1)
                    x=x.transpose(0,3,1,2)
                elif 'SpS'==preprocesser:
                    x,_=SpatialSmoothing()(x.transpose(0,2,3,1))
                    x=np.clip(x, 0, 1)
                    x=x.transpose(0,3,1,2)
                elif 'WebpF'==preprocesser:
                    x,_=defend_webpf_wrap(x.transpose(0,2,3,1))
                    x=x.transpose(0,3,1,2)
                elif 'RDG'==preprocesser:
                    x,_=defend_rdg_wrap(x.transpose(0,2,3,1))
                    x=x.transpose(0,3,1,2)
                elif 'FD'==preprocesser:
                    x,_=defend_fd_wrap(x.transpose(0,2,3,1))
                    x=x.transpose(0,3,1,2)
                elif 'BDR'==preprocesser:
                    x,_=defend_bdr_wrap(x.transpose(0,2,3,1))
                    x=x.transpose(0,3,1,2)
                elif 'SHIELD'==preprocesser:
                    x,_=defend_shield_wrap(x.transpose(0,2,3,1))
                    x=x.transpose(0,3,1,2)
                elif 'FD_ago'==preprocesser:
                    x,_=defend_FD_ago_warp(x.transpose(0,2,3,1))
                    x=x.transpose(0,3,1,2)
                else:
                    print('Wrong preprocessor: %s'%preprocesser)
            
            # 图像分类
            x           = self.trans_classifier(torch.from_numpy(x))
            y_pred      = self.classifier(x)
            y_pred      = F.softmax(y_pred,dim=1)
            _,label_classifier = torch.max(y_pred.data,dim=1)
            
            # 标签汇总
            idx_adv     = np.in1d(label_spectrum_before,cls_adv)
            idx_rpl     = idx_adv
            if self.flag_detect:
                idx_rpl=[]
            
            label_classifier[idx_rpl] = self.adv_cls_label
        torch.cuda.empty_cache()
        return label_classifier.detach().cpu().numpy()

def my_pred(attacked_img,labels,batch_size,method,spectrum_classifier_type,preprocesser):
    batch_num       = int(attacked_img.shape[0]/batch_size) 
    label_pred_list=[]
    for i in range(batch_num):
        start_idx=batch_size*i
        end_idx=min(batch_size*(i+1),attacked_img.shape[0])   
        label_pred_tmp = model(attacked_img[start_idx:end_idx,...],method,spectrum_classifier_type,preprocesser)
        label_pred_list.append(label_pred_tmp)
    label_pred=np.hstack(label_pred_list)    
    label_adv  = np.ones_like(label_pred)
    if 0 == method or 1==method:                   
        label_adv  = -1*np.ones_like(label_pred)                    
    
    correct,correct_adv,correct_pred = get_accuracy(label_pred, labels, label_adv, model.adv_cls_label)
    correct=correct/len(labels)
    correct_adv=correct_adv/len(labels)
    correct_pred=correct_pred/len(labels)
    return correct,correct_adv,correct_pred 
    
if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=3:
        print('Manual Mode !!!')
        model_vanilla_type = 'vgg16_imagenet'
        method = 2 # 0 原始对抗样本   1 低通滤波的对抗样本   2 低通滤波+分类的对抗样本 3 分类的对抗样本
        # device = 3
    else:
        print('Terminal Mode !!!')
        # model_spectrum_dir  = sys.argv[1]
        model_vanilla_type  = sys.argv[1]
        method         = int(sys.argv[2])
        # device         = int(sys.argv[4]) 
    method = 2 # 0 原始对抗样本   1 低通滤波的对抗样本   2 低通滤波+分类的对抗样本 3 分类的对抗样本
    device = 3
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    model_spectrum_dir = '../saved_tests/img_attack/spectrum_label/'+model_vanilla_type+'/checkpoint.pth.tar'
    adaboost_pkl = '../saved_tests/img_attack/spectrum_label/'+model_vanilla_type+'/'+model_vanilla_type+'_adaboost.pkl'
    svm_pkl='../saved_tests/img_attack/spectrum_label/'+model_vanilla_type+'/'+model_vanilla_type+'_svm.pkl'
    
    '''
    加载cifar-10图像
    '''
    # os.environ['CUDA_VISIBLE_DEVICES']=str(1)
    g.setup_seed(0)
    # dir_cifar     = g.dir_cifar
    # images,labels = load_CIFAR_batch(os.path.join(dir_cifar,'test_batch'))
    if 'imagenet' in model_vanilla_type:
        dataset='imagenet'
    else:
        dataset='cifar-10'
    if 'cifar-10'==dataset:
        dir_cifar     = g.dir_cifar
        images,labels = load_CIFAR_batch(os.path.join(dir_cifar,'test_batch'))
    elif 'imagenet'==dataset:
        with open(g.dir_feature_imagenet) as f:
            features=json.load(f)
        data_dir=os.path.join(g.dir_imagenet,'val')
        images,labels=load_imagenet_filenames(data_dir,features)
    
    
    '''
    加载模型
    '''
    model = proactiver_and_reactiver(model_spectrum_dir, adaboost_pkl, svm_pkl, model_vanilla_type, dataset)
    # with open(os.path.join("../models/cifar-10_class_to_idx.json")) as f:
    #     features=json.load(f)
    
    '''
    读取数据
    '''  
    batch           = g.pred_batch
     
    spectrums_list  = []
    labels_list     = []
    
    '''
    攻击与防护
    '''
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    attacks=['Deepfool_L2_IDP']#'CW_L2_IDP','Deepfool_L2_IDP',,'CW_Linf_IDP'
    eps_L2=[0.1,0.5,1.0,10.0,100.0]
    eps_Linf=[0.005,0.01,0.1,1.0,10.0]
    spectrum_classifiers=['cnn','adb','svm']
    preprocessers=['LPF1','LPF2','JPG','GuA','SpS','WebpF','RDG','FD','BDR','SHIELD','FD_ago']
    
    # attacks=['CW_L2_IDP']
    # eps_L2=[100.0]
    # eps_Linf=[0.005,10.0]
    # spectrum_classifiers=['cnn','adb','svm']
    # preprocessers=['LPF1','FD_ago']
    
    accuracys_cnn = np.zeros((len(attacks),len(eps_L2)))
    fprint_list=[]
    file_log=os.path.join(saved_dir,'result_log_cd.txt')
    logging.basicConfig(filename=file_log,
                    level=logging.FATAL)
    logging.fatal(('\n----------my defense result-----------'))
    
    results=np.zeros([len(attacks),len(eps_L2),len(spectrum_classifiers),len(preprocessers),3])
    for attack_idx,attack in enumerate(attacks):
        if 'L2' in attack:
            eps_now = eps_L2
        else:
            eps_now=eps_Linf
            
        images_now=images
        labels_now=labels            
            
        print('\n')
        for eps_idx,eps in enumerate(eps_now):
            if ((('CW_L2_IDP'==attack) or ('Deepfool_L2_IDP'==attack)) and (0 != eps_idx)):
                continue
            
            batch_now=batch
            if ((('CW_L2_IDP'==attack) or ('Deepfool_L2_IDP'==attack)) and ('imagenet' == dataset)):
                images_now=images[0:int(len(images)/10)]
                labels_now=labels[0:int(len(labels)/10)]
                batch_now=int(batch/10)
                # continue
                                 
            attacked_img = get_attacked_img(model.fmodel,images_now,labels_now,batch_now,attack,eps,dataset)
            
            for sc_idx,spectrum_classifier in enumerate(spectrum_classifiers):
                for pre_idx,preprocesser in enumerate(preprocessers):
                    correct,correct_adv,correct_pred=my_pred(attacked_img,labels_now,batch_now,method,spectrum_classifier,preprocesser)
                
                    prt_info='[ATTACK]:%s  [Eps]:%.3f  [%s] [%s] Acc:%.3f  Acc_pred:%.3f  Acc_adv:%.3f'%(attack,eps,spectrum_classifier,preprocesser, 100*correct,100*correct_pred,100*correct_adv)
                    print(prt_info)
                    logging.fatal(prt_info)
                    results[attack_idx,eps_idx,sc_idx,pre_idx,0]=correct
                    results[attack_idx,eps_idx,sc_idx,pre_idx,1]=correct_pred
                    results[attack_idx,eps_idx,sc_idx,pre_idx,2]=correct_adv
    np.save(os.path.join(saved_dir,'result_log_cd.npy'),results)
