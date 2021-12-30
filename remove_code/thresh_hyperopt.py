#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:24:38 2021

@author: estar
"""

from hyperopt import hp, fmin, rand, space_eval, Trials
from tqdm import tqdm
from defense_ago import Cal_channel_wise_qtable
from art.attacks.evasion import FastGradientMethod

import numpy as np
import torch
import os 
import sys
import torch.nn as nn
import pickle
import cv2
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from adaptivce_defense import adaptive_defender
import matplotlib.pyplot as plt
sys.path.append('../common_code')
import general as g
import pickle

def get_acc(fmodel,images,labels):
    predictions = fmodel.predict(images)
    predictions = np.argmax(predictions,axis=1)
    cors = np.sum(predictions==labels)
    return cors

def get_defended_attacked_acc(fmodel,dataloader,attackers,defenders,defender_names):
    cors=np.zeros((len(attackers)+1,len(defenders)+1))
    for i, (images, labels) in enumerate(dataloader):
        images=images.numpy()
        labels=labels.numpy()
        for j in range(len(attackers)+1):
            images_att=images.copy()
            eps=0
            if j>0:
                try:
                    eps=attackers[j-1].eps
                except:
                    eps=0
                images_att  = attackers[j-1].generate(x=images.copy())
            for k in range(len(defenders)+1):
                    images_def = images_att.copy()
                    if k>0:
                        if 'ADAD-flip'==defender_names[k-1]:
                            images_def,_ = defenders[k-1](images_att.transpose(0,2,3,1).copy(),labels.copy(),None,0)
                        elif 'ADAD+eps-flip'==defender_names[k-1]:
                            images_def,_ = defenders[k-1](images_att.transpose(0,2,3,1).copy(),labels.copy(),eps*np.ones(images_att.shape[0]),0)
                        else:
                            images_def,_ = defenders[k-1](images_att.transpose(0,2,3,1).copy(),labels.copy())
                        images_def=images_def.transpose(0,3,1,2)
                    cors[j,k] += get_acc(fmodel,images_def,labels)
    cors=cors/len(dataloader.dataset)
    return cors

def get_shapleys_batch_adv(attack, model, dataloader, num_samples):
    
    dataiter = iter(dataloader)
    
    images = []
    images_adv = []
    num_samples_now = 0
    for i in range(len(dataloader)):
        # t.set_description("Get attacked samples {0:3d}".format(num_samples_now))
        data, label = dataiter.next()
        
        save_cln = data.detach().numpy()
        save_adv = attack.generate(save_cln)
        
        images.append(save_cln)
        images_adv.append(save_adv)
        
        num_samples_now=num_samples_now+save_cln.shape[0]
        torch.cuda.empty_cache()
        if num_samples_now>=num_samples:
            break    

    if num_samples_now<num_samples:
        print('\n!!! not enough samples\n')
    
    images_np=None
    images_adv_np=None
    if len(images)>0:
        images_np=np.vstack(images)
    if len(images_adv)>0:
        images_adv_np=np.vstack(images_adv)
    return images_np,images_adv_np

def cal_table(threshs,saved_dir,fmodel,model,attacker_name,img_num,eps):
        
    table_dict=dict()
    table_dict[0]=np.ones([8,8,3])
    for eps_now in eps:
        attacker,_=g.select_attack(fmodel,attacker_name,eps_now)
        
        clean_imgs,adv_imgs=get_shapleys_batch_adv(attacker,model,dataloader,img_num)
        
        clean_imgs=np.transpose(clean_imgs.copy(),(0,2,3,1))*255
        adv_imgs=np.transpose(adv_imgs.copy(),(0,2,3,1))*255
        clean_imgs_ycc=g.rgb_to_ycbcr(clean_imgs)
        adv_imgs_ycc=g.rgb_to_ycbcr(adv_imgs)
        
        np.set_printoptions(suppress=True)
        a_qtable,_,_,_=Cal_channel_wise_qtable(clean_imgs_ycc, adv_imgs_ycc,threshs)
        a_qtable=np.round(a_qtable)
        table_dict[eps_now]=a_qtable
    # print(table_dict[0.5])
    pickle.dump(table_dict, open(os.path.join(saved_dir,'table_dict.pkl'),'wb'))
       
def objective(args):
    
    threshs=np.array((args[0],args[1],args[2]))
    saved_dir=args[3]
    fmodel=args[4]
    model=args[5]
    attacker_name=args[6]
    img_num=args[7]
    eps=args[8]
    
    print(threshs)
    '''
    计算量表并初始化防御
    '''
    cal_table(threshs,saved_dir,fmodel,model,attacker_name,img_num,eps)

    table_pkl=os.path.join(saved_dir,'table_dict.pkl')
    defender=adaptive_defender(table_pkl,None,None)
  
    '''
    计算防御效果
    '''            
    # 标为原始样本
 
    accs=get_defended_attacked_acc(fmodel,dataloader,attacks,[defender.defend],['ADAD+eps-flip'])
    metric=accs.mean(axis=0)[1]
    output=-metric
    print(accs[:,1])
    return output

if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=2:
        print('Manual Mode !!!')
        model_vanilla_type    = 'allconv' 
    else:
        print('Terminal Mode !!!')
        model_vanilla_type  = str(sys.argv[1])
       
    # global fmodel,model,attacker,attacker_name,img_num,eps
    attacker_name='FGSM_L2_IDP'
    img_num=1000
    max_evals=100
    resolution=0.01
    
    saved_dir = '../saved_tests/img_attack/accuracy/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        
    '''
    加载模型
    '''
    dir_model  = '../models/cifar_vanilla_'+model_vanilla_type+'.pth.tar'
    model,dataset_name=g.select_model(model_vanilla_type, dir_model)
    model.eval()
    
        
    '''
    加载图像
    '''
    data_setting=g.dataset_setting(dataset_name)
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,'val')
    dataloader = DataLoader(dataset, batch_size=data_setting.pred_batch_size, drop_last=False)   
    
    fmodel = PyTorchClassifier(model = model,nb_classes=data_setting.nb_classes,clip_values=(0,1),
                               input_shape=data_setting.input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(data_setting.mean, data_setting.std))
    
    '''
    攻击初始化
    '''
    attacks=[]
    attack_names=[]
    eps=[0.001,0.1,0.5,1.0,10.0]
    for i in range(len(eps)):
          attacks.append(FastGradientMethod(estimator=fmodel,eps=eps[i],norm=2,eps_step=eps[i],batch_size=data_setting.pred_batch_size))
          attack_names.append('FGSM_L2_'+str(eps[i]))    
    
    '''
    超参数优化
    '''
    trials=Trials()
    # space =[hp.quniform('t0',0,0.5,resolution),hp.quniform('t1',0,1,resolution),hp.quniform('t2',0,1,resolution)]
    space =[hp.quniform('t0',0,1.0,resolution),hp.quniform('t1',0,1,resolution),hp.quniform('t2',0,1,resolution),#hp.choice('t0',[0.9]),hp.choice('t1',[0.01]),hp.choice('t2',[0.01])
            hp.choice('saved_dir',[saved_dir]),
            hp.choice('fmodel',[fmodel]),
            hp.choice('model',[model]),
            hp.choice('attacker_name',[attacker_name]),
            hp.choice('img_num',[img_num]),
            hp.choice('eps',[eps])]
    
    best=fmin(objective,space,algo=rand.suggest,max_evals=max_evals,verbose=True, max_queue_len=3,trials=trials)
    pickle.dump(trials,open(os.path.join(saved_dir,'hyperopt_trail.pkl'),"wb"))
    trials=pickle.load(open(os.path.join(saved_dir,'hyperopt_trail.pkl'),"rb"))
    print(best)
    
    '''
    可视化
    '''
    trials_list=[]
    parameters=['t0','t1','t2']
    cols = len(parameters)
    f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        trials_list.append(np.expand_dims(np.vstack((xs,ys)),axis=0))
        axes[i].scatter(
            xs,
            ys,
            s=20,
            linewidth=0.01,
            alpha=0.25,
            c=cmap(float(i) / len(parameters)))
        axes[i].set_title(val)
        axes[i].set_ylim([0.0, 1.0])
    plt.savefig(os.path.join(saved_dir,'hyperopt_trail.png'), bbox_inches='tight')
    trials_np=np.vstack(trials_list)
    np.save(os.path.join(saved_dir,'hyperopt_trail_np.npy'),trials_np)
    print(trials.best_trial)
    
    cal_table([best['t0'],best['t1'],best['t2']],saved_dir,fmodel,model,attacker_name,img_num,eps)