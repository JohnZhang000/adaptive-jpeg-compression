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
from my_classifier import Net as spectrum_net
from my_spectrum_analyzer import img_spectrum_analyzer
from train_code.my_img_transformer import img_transformer
from tqdm import tqdm

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import joblib
import logging

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

def select_attack(fmodel, att_method, eps):
    if att_method   == 'FGSM_L2_IDP':
        attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=2)
    elif att_method == 'PGD_L2_IDP':
        attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=2,batch_size=512,verbose=False)
    elif att_method == 'CW_L2_IDP':
        attack = CarliniL2Method(classifier=fmodel,batch_size=512,verbose=False)
    elif att_method == 'Deepfool_L2_IDP':
        attack = DeepFool(classifier=fmodel,batch_size=512,verbose=False)
        
    elif att_method == 'FGSM_Linf_IDP':
        attack = FastGradientMethod(estimator=fmodel,eps=eps,norm=np.inf)
    elif att_method == 'PGD_Linf_IDP':
        attack = ProjectedGradientDescent(estimator=fmodel,eps=eps,norm=np.inf,batch_size=512,verbose=False)
    elif att_method == 'CW_Linf_IDP':
        attack = CarliniLInfMethod(classifier=fmodel,eps=eps,batch_size=512,verbose=False)
    
    elif att_method == 'FGSM_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
    elif att_method == 'PGD_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
    elif att_method == 'CW_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='carlini',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
    elif att_method == 'Deepfool_L2_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='deepfool',attacker_params={'eps':eps,'norm':2,'verbose':False},eps=eps,norm=2,batch_size=512,verbose=False)
    
    elif att_method == 'FGSM_Linf_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='fgsm',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=512,verbose=False)
    elif att_method == 'PGD_Linf_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='pgd',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=512,verbose=False)
    elif att_method == 'CW_Linf_UAP':
        attack = UniversalPerturbation(classifier=fmodel,attacker='carlini_inf',attacker_params={'eps':eps,'norm':np.inf,'verbose':False},eps=eps,norm=np.inf,batch_size=512,verbose=False)
    
    else:
        raise Exception('Wrong Attack Mode: {} !!!'.format(att_method))
    return attack, eps
    
def get_attacked_img(fmodel, images, labels, attack_idx, eps_idx):
    attack,eps=select_attack(fmodel, attack_idx,eps_idx)
    images_adv = attack.generate(x=images,y=labels)
    return images_adv       
 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class proactiver_and_reactiver(nn.Module):
    def __init__(self,spectrum_ckpt, adaboost_pkl, svm_pkl, classifier_type, dataset):
        super(proactiver_and_reactiver,self).__init__()
        if 'imagenet'==dataset:
            self.img_size=224
            mean_now=[0.485, 0.456, 0.406]
            std_now=[0.229, 0.224, 0.225]
            self.num_classes=1000
            self.spectrum_size=0
            self.spectrum_mean_std=[0,0]
            self.adv_cls_label=1001
            
        elif 'cifar-10'==dataset:
            self.img_size=32
            mean_now=[0.5] * 3
            std_now=[0.5] * 3
            self.num_classes=10
            self.spectrum_size=16
            self.spectrum_mean_std=[0.0, 1.0]
            self.fft_transformer = img_transformer(8,0,6)
            self.adv_cls_label=11
        else:
            print('ERROR DATASET')
        
        self.s_analyzer=img_spectrum_analyzer(self.img_size).batch_get_spectrum_feature

        self.spectrum_net = spectrum_net(self.spectrum_size)
        checkpoint = torch.load(spectrum_ckpt)
        self.spectrum_net.load_state_dict(checkpoint['state_dict'])
        self.spectrum_net.cuda().eval()
        
        if adaboost_pkl:
            self.adabooster=joblib.load(adaboost_pkl)
        if svm_pkl:
            self.svmer=joblib.load(svm_pkl)
        
        if classifier_type == 'allconv':
            self.classifier = AllConvNet(10).eval()
            dir_model  = '../models/cifar_vanilla_allconv.pth.tar'
        elif classifier_type == 'densenet':
            self.classifier = densenet(num_classes=10).eval()
            dir_model  = '../models/cifar_vanilla_densenet.pth.tar'
        elif classifier_type == 'wrn':
            self.classifier = WideResNet(40, 10, 2, 0.0).eval()
            dir_model  = '../models/cifar_vanilla_wrn.pth.tar'
        elif classifier_type == 'resnext':
            self.classifier = resnext29(num_classes=10).eval()
            dir_model  = '../models/cifar_vanilla_resnext.pth.tar' 
        else:
            print('Error model name!!!')
            
        self.trans_classifier=transforms.Compose([transforms.Normalize(mean=mean_now, std=std_now)])     
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()
        checkpoint = torch.load(dir_model)
        self.classifier.load_state_dict(checkpoint['state_dict'])
        
        mean   = np.array((0.5,0.5,0.5),dtype=np.float32)
        std    = np.array((0.5,0.5,0.5),dtype=np.float32)
        self.fmodel = PyTorchClassifier(model = self.classifier,nb_classes=10,clip_values=(0,1),
                                   input_shape=(3,32,32),loss = nn.CrossEntropyLoss(),
                                   preprocessing=(mean, std)) 
        
    def forward(self,x, method, spectrum_classifier):
        
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
            if 0 != method:
                x      = self.fft_transformer.img_transform_tc(x)
            
            # 图像分类
            x           = self.trans_classifier(torch.from_numpy(x))
            y_pred      = self.classifier(x)
            y_pred      = F.softmax(y_pred,dim=1)
            _,label_classifier = torch.max(y_pred.data,dim=1)
            
            # 标签汇总
            idx_adv     = np.in1d(label_spectrum_before,cls_adv)
            idx_rpl     = idx_adv
            if 2 != method:
                idx_rpl=[]
            
            label_classifier[idx_rpl] = self.adv_cls_label
        
        return label_classifier.detach().cpu().numpy()

def my_pred(attacked_img,method,spectrum_classifier_type):
    label_pred = model(attacked_img,method,spectrum_classifier_type)
    label_adv  = np.ones_like(label_pred)
    if 0 == method:                   
        label_adv  = -1*np.ones_like(label_pred)                    
    
    correct,correct_adv,correct_pred = get_accuracy(label_pred, labels_batch, label_adv, model.adv_cls_label)
    # correct_all += correct
    # correct_adv_all += correct_adv
    # correct_pred_all += correct_pred
    return correct,correct_adv,correct_pred 
    
if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=5:
        print('Manual Mode !!!')
        model_spectrum_dir = '../saved_tests/img_attack/spectrum_label/cifar-10/checkpoint.pth.tar'
        adaboost_pkl = '../saved_tests/img_attack/spectrum_label/cifar-10/cifar-10_adaboost.pkl'
        svm_pkl='../saved_tests/img_attack/spectrum_label/cifar-10/cifar-10_svm.pkl'
        model_vanilla_type = 'allconv'
        method = 2 # 0 原始对抗样本   1 低通滤波的对抗样本   2 低通滤波+分类的对抗样本
        device = 3
    else:
        print('Terminal Mode !!!')
        model_spectrum_dir  = sys.argv[1]
        model_vanilla_type  = sys.argv[2]
        method         = int(sys.argv[3])
        device         = int(sys.argv[4]) 
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    '''
    加载cifar-10图像
    '''
    os.environ['CUDA_VISIBLE_DEVICES']=str(device)
    setup_seed(0)
    dir_cifar     = '../../../../../media/ubuntu204/F/Dataset/Dataset_tar/cifar-10-batches-py'
    images,labels = load_CIFAR_batch(os.path.join(dir_cifar,'test_batch'))
    
    
    '''
    加载模型
    '''
    model = proactiver_and_reactiver(model_spectrum_dir, adaboost_pkl, svm_pkl, model_vanilla_type, 'cifar-10')
    with open(os.path.join("../models/cifar-10_class_to_idx.json")) as f:
        features=json.load(f)
    
    '''
    读取数据
    '''  
    batch           = 1000
     
    spectrums_list  = []
    labels_list     = []
    
    '''
    攻击与防护
    '''
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    attacks=['FGSM_L2_IDP','PGD_L2_IDP','CW_L2_IDP','Deepfool_L2_IDP',
             'FGSM_Linf_IDP','PGD_Linf_IDP','CW_Linf_IDP']
    eps_L2=[0.1,0.5,1.0,10.0,100.0]
    eps_Linf=[0.005,0.01,0.1,1.0,10.0]
    accuracys_cnn = np.zeros((len(attacks),len(eps_L2)))
    fprint_list=[]
    file_log=os.path.join(saved_dir,'result_log.txt')
    logging.basicConfig(filename=file_log,
                    level=logging.INFO)
    logging.info(('\n----------my defense result-----------'))
    
    for attack_idx,attack in enumerate(attacks):
        if 'L2' in attack:
            eps_now = eps_L2
        else:
            eps_now=eps_Linf
            
        images_now=images
        labels_now=labels            
            
        print('\n')
        for eps_idx,eps in enumerate(eps_now):
            if (((2==attack_idx) or (3==attack_idx)) and (0 != eps_idx)):
                continue
            correct_cnn=[]
            correct_adv_cnn=[]
            correct_pred_cnn=[]
            correct_adb=[]
            correct_adv_adb=[]
            correct_pred_adb=[]
            correct_svm=[]
            correct_adv_svm=[]
            correct_pred_svm=[]
            
            batch_num       = int(len(labels_now)/batch) 
            for i in range(batch_num):#tqdm(range(batch_num)):

                images_batch = images_now[batch*i:batch*(i+1),...].transpose(0,3,1,2)
                labels_batch = labels_now[batch*i:batch*(i+1),...]
                        
                attacked_img = get_attacked_img(model.fmodel,images_batch,labels_batch,attack,eps)
                
                correct,correct_adv,correct_pred=my_pred(attacked_img,method,'cnn')
                correct_cnn.append(correct)
                correct_adv_cnn.append(correct_adv)
                correct_pred_cnn.append(correct_pred)
                
                correct,correct_adv,correct_pred=my_pred(attacked_img,method,'adb')
                correct_adb.append(correct)
                correct_adv_adb.append(correct_adv)
                correct_pred_adb.append(correct_pred)
                
                correct,correct_adv,correct_pred=my_pred(attacked_img,method,'svm')
                correct_svm.append(correct)
                correct_adv_svm.append(correct_adv)
                correct_pred_svm.append(correct_pred)
                
            accuracy_all_cnn = np.array(correct_cnn).sum()/len(labels_now)
            accuracy_pred_all_cnn = np.array(correct_pred_cnn).sum()/len(labels_now)
            accuracy_adv_all_cnn = np.array(correct_adv_cnn).sum()/len(labels_now)
            # accuracys_cnn[attack_idx,eps_idx]=accuracy_all_cnn*100
            prt_info='[ATTACK]:%s  [Eps]:%.3f  [CNN] Acc:%.1f  Acc_adv:%.1f  Acc_pred:%.1f'%(attack,eps,100*accuracy_all_cnn,100*accuracy_adv_all_cnn,100*accuracy_pred_all_cnn)
            print(prt_info)
            logging.info(prt_info)
            
            accuracy_all_adb = np.array(correct_adb).sum()/len(labels_now)
            accuracy_pred_all_adb = np.array(correct_pred_adb).sum()/len(labels_now)
            accuracy_adv_all_adb = np.array(correct_adv_adb).sum()/len(labels_now)
            # accuracys_adb[attack_idx,eps_idx]=accuracy_all_adb*100
            prt_info='[ATTACK]:%s  [Eps]:%.3f  [ADB] Acc:%.1f  Acc_adv:%.1f  Acc_pred:%.1f'%(attack,eps,100*accuracy_all_adb,100*accuracy_adv_all_adb,100*accuracy_pred_all_adb)
            print(prt_info)
            logging.info(prt_info)
            
            accuracy_all_svm = np.array(correct_svm).sum()/len(labels_now)
            accuracy_pred_all_svm = np.array(correct_pred_svm).sum()/len(labels_now)
            accuracy_adv_all_svm = np.array(correct_adv_svm).sum()/len(labels_now)
            # accuracys_cnn[attack_idx,eps_idx]=accuracy_all_cnn*100
            prt_info='[ATTACK]:%s  [Eps]:%.3f  [SVM] Acc:%.1f  Acc_adv:%.1f  Acc_pred:%.1f'%(attack,eps,100*accuracy_all_svm,100*accuracy_adv_all_svm,100*accuracy_pred_all_svm)
            print(prt_info)
            logging.info(prt_info)
            # f=open(file_log,'a')
            # f.write(prt_info+'\n')
            # f.close()
            # fprint_list.append(prt_info)
            
