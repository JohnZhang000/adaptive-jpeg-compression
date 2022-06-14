#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:24:38 2021

@author: estar
"""
import gc
from hyperopt import hp, fmin, rand, Trials
# from hyperopt.mongoexp import MongoTrials

# from tqdm import tqdm
from adaptivce_defense import Cal_channel_wise_qtable,Cal_channel_wise_qtable_mp
from art.attacks.evasion import FastGradientMethod

import numpy as np
import torch
from torch.optim import Adam
import os 
import sys
import torch.nn as nn
import pickle
# import cv2
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from adaptivce_defense import adaptive_defender
import matplotlib.pyplot as plt
sys.path.append('../common_code')
import general as g
# import pickle

def forward_proprecess(x,mean,std):
    if isinstance(x,np.ndarray): x=torch.from_numpy(x)
    if len(x.shape)==3: x=x.unsqueeze(0)
    x=(x-mean.reshape(1,-1, 1, 1))/std.reshape(1,-1, 1, 1)
    x=x.requires_grad_(True).cuda()
    return x

def inverse_proprecess(x,mean,std):
    x=x.squeeze(0).detach().cpu()#.permute(1,2,0).numpy()
    x=x*std.reshape(-1, 1, 1)+mean.reshape(-1, 1, 1)
    return x

def BPDA(model,x_orig,y_orig,mean,std,classes=10,epoch=3,defend=None,saved_dir=None):
    acc=[]
    att=[]
    L2s=[]
    # x_adv = x_orig.clone()
    x_adv_bpda = x_orig.clone()
    y_adv = list(range(classes))
    y_adv.remove(y_orig.cpu())
    y_adv = np.random.choice(y_adv)
    for i in range(epoch):
        if defend:  
            x_adv_def,_=defend(x_adv_bpda)
            if len(x_adv_def.shape)==4 and x_adv_def.shape[1]==x_adv_def.shape[2]: x_adv_def=torch.from_numpy(x_adv_def).squeeze(0).permute(2,0,1)
        else:
            x_adv_def=x_adv_bpda
        x_orig_tc=forward_proprecess(x_orig,mean,std)
        x_adv_def_tc=forward_proprecess(x_adv_def,mean,std)
        x_adv_bpda_tc=forward_proprecess(x_adv_bpda,mean,std)

        p,g,L2=loss_grad_bpda(model,x_orig_tc,x_adv_bpda_tc,x_adv_def_tc,y_adv,mean,std)
        x_adv_bpda_tc -= LR * g
        x_adv_bpda = inverse_proprecess(x_adv_bpda_tc,mean,std)
        x_adv_bpda = torch.clip(x_adv_bpda, 0, 1)
        if p!=y_orig:# and L2<EPSILON:
            acc.append(0)
        else:
            acc.append(1)
        if p==y_adv:
            att.append(1)
        else:
            att.append(0)
        L2s.append(L2)
    return np.array(acc).reshape(1,-1),np.array(att).reshape(1,-1),np.array(L2s).reshape(1,-1)

def get_acc_mp(model,mean,std,images,labels):
    images=(images.transpose(0,2,3,1)-mean)/std
    images=torch.from_numpy(images.transpose(0,3,1,2)).cuda()
    with torch.no_grad():
        predictions = model(images)
    predictions = np.argmax(predictions.cpu().numpy(),axis=1)
    cors = np.sum(predictions==labels)
    return cors

def get_acc(fmodel,images,labels):
    with torch.no_grad():
        predictions = fmodel.predict(images)
    predictions = np.argmax(predictions,axis=1)
    cors = np.sum(predictions==labels)
    return cors

def get_defended_attacked_acc_per_batch(model,mean,std,attack_eps,defenders,defender_names,imgs_in,labels_in):
    cors=np.zeros((len(attack_eps)+1,len(defenders)+1))
    for i in range(imgs_in.shape[0]):
            images_att=imgs_in[i,...].copy()
            labels=labels_in
            for k in range(len(defenders)+1):
                    images_def = images_att.copy()
                    if k>0:
                        if 'ADAD-flip'==defender_names[k-1]:
                            images_def,_ = defenders[k-1](images_def.transpose(0,2,3,1).copy(),labels.copy(),None,0)
                        elif 'ADAD+eps-flip'==defender_names[k-1]:
                            images_def,_ = defenders[k-1](images_def.transpose(0,2,3,1).copy(),labels.copy(),attack_eps[i]*np.ones(images_def.shape[0]),0)
                        else:
                            images_def,_ = defenders[k-1](images_def.transpose(0,2,3,1).copy(),labels.copy())
                        images_def=images_def.transpose(0,3,1,2)
                    images_def_cp = images_def.copy()
                    cors[i,k] += get_acc_mp(model,mean,std,images_def_cp,labels)
                    del images_def,images_def_cp
    cors=cors/imgs_in.shape[1]
    return np.expand_dims(cors,axis=0)

def get_defended_attacked_acc_mp(fmodel,attack_eps,defenders,defender_names,imgs_in,labels_in,batch_size):
    model=fmodel.model
    mean=fmodel.preprocessing.mean
    std=fmodel.preprocessing.std
    
    # start pool
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(data_setting.device_num*2)

    # start_idx=0
    # end_idx=batch_size
    # get_defended_attacked_acc_per_batch(model,mean,std,attack_eps,defenders,defender_names,imgs_in[:,start_idx:end_idx,...].copy(),labels_in[start_idx:end_idx])
    
    batch_num=int(np.ceil(imgs_in.shape[1]/batch_size))
    pool_list=[]
    for j in range(batch_num):
        start_idx=j*batch_size
        end_idx=min((j+1)*batch_size,imgs_in.shape[1])
        res=pool.apply_async(get_defended_attacked_acc_per_batch,
                        args=(model,mean,std,attack_eps,defenders,defender_names,imgs_in[:,start_idx:end_idx,...].copy(),labels_in[start_idx:end_idx]))
    pool_list.append(res)
    pool.close()
    pool.join()

    corss=[]
    for i in pool_list:
            cors = i.get()
            corss.append(cors)
    cors_np=np.vstack(corss).sum(axis=0)
    # cors=cors_np/len(dataloader.dataset)
    return cors_np

def get_defended_attacked_acc(fmodel,attack_eps,defenders,defender_names,imgs_in,labels_in,batch_size):
    cors=np.zeros((imgs_in.shape[0],len(defenders)))
    
    batch_num=int(np.ceil(imgs_in.shape[1]/batch_size))
    for i in range(imgs_in.shape[0]):
        for j in range(batch_num):
            start_idx=j*batch_size
            end_idx=min((j+1)*batch_size,imgs_in.shape[1])
            images_att=imgs_in[i,start_idx:end_idx,...].copy()
            labels=labels_in[start_idx:end_idx]
            for k in range(len(defenders)):
                    images_def = images_att.copy()
                    images_def,_ = defenders[k](images_def.transpose(0,2,3,1).copy(),labels.copy(),attack_eps[i]*np.ones(images_def.shape[0]),0)
                    images_def=images_def.transpose(0,3,1,2)
                    images_def_cp=images_def.copy()
                    cors[i,k] += get_acc(fmodel,images_def_cp,labels)
    cors=cors/imgs_in.shape[1]
    return cors

def get_shapleys_batch_adv(attack, dataloader, num_samples):
    
    dataiter = iter(dataloader)
    
    images = []
    images_adv = []
    labels = []
    num_samples_now = 0
    for i in range(len(dataloader)):
        # t.set_description("Get attacked samples {0:3d}".format(num_samples_now))
        data, label = dataiter.next()
        
        save_cln = data.detach().numpy()
        save_adv = attack.generate(save_cln)
        
        images.append(save_cln)
        images_adv.append(save_adv)
        labels.append(label)
        
        num_samples_now=num_samples_now+save_cln.shape[0]
        torch.cuda.empty_cache()
        if num_samples_now>=num_samples:
            break    

    if num_samples_now<num_samples:
        try:
            print('\n!!! not enough samples for eps %.1f\n'%attack.eps)
        except:
            print('\n!!! not enough samples \n')
            
    
    images_np=None
    images_adv_np=None
    labels_np=None
    if len(images)>0:
        images_np=np.vstack(images)
    if len(images_adv)>0:
        images_adv_np=np.vstack(images_adv)
    if len(labels)>0:
        labels_np=np.hstack(labels)
    return images_np,images_adv_np,labels_np

def cal_table(threshs,saved_dir,cln_imgs_in,adv_imgs_in,attack_eps):
        
    table_dict=dict()
    table_dict[0]=np.ones([8,8,3])
    for i in range(adv_imgs_in.shape[0]):
        
        clean_imgs_ct=cln_imgs_in.copy()
        adv_imgs_ct=adv_imgs_in[i,...].copy()
        
        clean_imgs=np.transpose(clean_imgs_ct,(0,2,3,1))*255
        adv_imgs=np.transpose(adv_imgs_ct,(0,2,3,1))*255
        clean_imgs_ycc=g.rgb_to_ycbcr(clean_imgs)
        adv_imgs_ycc=g.rgb_to_ycbcr(adv_imgs)
        
        np.set_printoptions(suppress=True)
        a_qtable,_,_,_=Cal_channel_wise_qtable(clean_imgs_ycc, adv_imgs_ycc,threshs)
        a_qtable=np.round(a_qtable)
        table_dict[attack_eps[i]]=a_qtable
        del clean_imgs,adv_imgs,clean_imgs_ycc,adv_imgs_ycc
        gc.collect()
    # print(table_dict[0.5])
    pickle.dump(table_dict, open(os.path.join(saved_dir,'table_dict_'+str(attack_eps[0])+'.pkl'),'wb'))
    
# def cal_table_jpeg(threshs,saved_dir,cln_imgs_in,adv_imgs_in,attack_eps):
        
#     table_dict=dict()
#     table_dict[0]=np.ones([8,8,3])
#     for i in range(adv_imgs_in.shape[0]):
#         a_qtable=np.ones([8,8,3])
#         a_qtable[:,:,0]=g.scale_table(g.table_y,threshs[i]*100)
#         a_qtable[:,:,1]=g.scale_table(g.table_c,threshs[i]*100)
#         a_qtable[:,:,2]=g.scale_table(g.table_c,threshs[i]*100)
#         table_dict[attack_eps[i+1]]=a_qtable
#     pickle.dump(table_dict, open(os.path.join(saved_dir,'table_dict.pkl'),'wb'))
       
def objective(args):
    
    threshs=np.array((args[0],args[1],args[2]))
    saved_dir=args[3]
    # fmodel=args[4]
    # model=args[5]
    cln_imgs_in=args[4]
    adv_imgs_in=args[5]
    labels=args[6]
    batch_size=args[7]
    nb_classes=args[8]
    input_size=args[9]
    pred_batch_size=args[10]
    attack_eps=[args[11]]
    fmodel=args[12]
    
    # print(threshs)
        
    '''
    计算量表
    '''
    cal_table(threshs,saved_dir,cln_imgs_in,adv_imgs_in,attack_eps)

    table_pkl=os.path.join(saved_dir,'table_dict_'+str(attack_eps[0])+'.pkl')
    defender=adaptive_defender(table_pkl,None,nb_classes,input_size,pred_batch_size,None)
    
    '''
    计算防御效果
    '''            
    # 标为原始样本
    imgs_in=adv_imgs_in
    labels_in=labels
    accs=get_defended_attacked_acc(fmodel,attack_eps,[defender.defend],['ADAD+eps-flip'],imgs_in,labels_in,batch_size)
    metric=accs.mean(axis=0)[0]
    output=-metric
    # print(accs)#[:,1])
    return output

def img_and_model_init(model_vanilla_type):
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
    dataset=g.load_dataset(dataset_name,data_setting.dataset_dir,'val',data_setting.hyperopt_img_val_num)
    dataloader = DataLoader(dataset, batch_size=data_setting.pred_batch_size, drop_last=False, shuffle=False, num_workers=data_setting.workers, pin_memory=True)   
    
    optimizer=Adam
    optimizer.state_dict
    fmodel = PyTorchClassifier(model = model,nb_classes=data_setting.nb_classes,clip_values=(0,1),
                               input_shape=data_setting.input_shape,loss = nn.CrossEntropyLoss(),
                               preprocessing=(data_setting.mean, data_setting.std),
                               optimizer=optimizer)
    return data_setting,dataloader,fmodel

def attack_init(fmodel,dataloader,data_setting):
    '''
    攻击初始化
    '''
    attacks=[]
    attack_names=[]
    attack_name='FGSM_L2_IDP'
    eps=[]#[0.1,0.5,1.0]
    eps.append(0.1*data_setting.eps_L2[0])
    for i in range(len(data_setting.eps_L2)):
        # eps.append(data_setting.eps_L2[i]*0.9)
        eps.append(data_setting.eps_L2[i]*1.0)
        # eps.append(data_setting.eps_L2[i]*1.1)
    # eps=[10.0,1.0,0.5,0.1]
    for i in range(len(eps)):
           # attacks.append(FastGradientMethod(estimator=fmodel,eps=eps[i],norm=2,eps_step=eps[i],batch_size=data_setting.pred_batch_size))
           attack_names.append(attack_name+'_'+str(eps[i]))  
           attacker,_=g.select_attack(fmodel,attack_name,eps[i])
           attacks.append(attacker)          
        
    adv_imgs_list=[]
    for i in range(len(attacks)):
        attacker=attacks[i]
        clean_imgs,adv_imgs_tmp,labels=get_shapleys_batch_adv(attacker,dataloader,data_setting.hyperopt_img_num)
        adv_imgs_list.append(np.expand_dims(adv_imgs_tmp,axis=0))
    adv_imgs=np.vstack(adv_imgs_list)
    del adv_imgs_tmp,adv_imgs_list
    gc.collect()
    return clean_imgs,adv_imgs,labels,eps
    

if __name__=='__main__':    
    '''
    settings
    '''
    # 配置解释器参数
    if len(sys.argv)!=2:
        print('Manual Mode !!!')
        model_vanilla_type    = 'vgg16_imagenet' 
    else:
        print('Terminal Mode !!!')
        model_vanilla_type  = str(sys.argv[1])
       
    # global fmodel,model#,attacker#,attacker_name,img_num,eps
    # attacker_name='FGSM_L2_IDP'
    # max_evals=4
    # resolution=0.01
    
    g.setup_seed(0)
    
    saved_dir = '../saved_tests/BPDA/'+model_vanilla_type
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        
    '''
    初始化
    '''
    data_setting,dataloader,fmodel=img_and_model_init(model_vanilla_type)
    
    '''
    生成图像
    '''
    clean_imgs,adv_imgs,labels,eps=attack_init(fmodel, dataloader, data_setting)
    
    '''
    超参数优化
    '''
    for idx_eps,eps_now in enumerate(eps):
        print('Hyperopt thresh for {}'.format(eps_now))
        trials=Trials()
        space =[
                # hp.choice('t0',[0.066]),hp.choice('t1',[0.003]),hp.choice('t2',[0.165]),
                hp.quniform('t0',data_setting.hyperopt_thresh_lower,data_setting.hyperopt_thresh_upper,data_setting.hyperopt_resolution),
                hp.quniform('t1',data_setting.hyperopt_thresh_lower,data_setting.hyperopt_thresh_upper,data_setting.hyperopt_resolution),
                hp.quniform('t2',data_setting.hyperopt_thresh_lower,data_setting.hyperopt_thresh_upper,data_setting.hyperopt_resolution),#hp.choice('t0',[0.9]),hp.choice('t1',[0.01]),hp.choice('t2',[0.01])
                hp.choice('saved_dir',[saved_dir]),
                
                hp.choice('clean_imgs',[clean_imgs]),
                hp.choice('adv_imgs_in',[np.expand_dims(adv_imgs[idx_eps,...],axis=0)]),
                hp.choice('labels',[labels]),
                
                hp.choice('batch_size',[data_setting.pred_batch_size]),
                hp.choice('nb_classes',[data_setting.nb_classes]),
                hp.choice('input_size',[data_setting.input_shape[-1]]),
                hp.choice('pred_batch_size',[data_setting.pred_batch_size]),
                hp.choice('attack_eps',[eps_now]),
                hp.choice('fmodel',[fmodel])]
        
        best=fmin(objective,space,algo=rand.suggest,max_evals=data_setting.hyperopt_max_evals,verbose=True, max_queue_len=1,trials=trials)
        pickle.dump(trials,open(os.path.join(saved_dir,'hyperopt_trail_'+str(eps_now)+'.pkl'),"wb"))
        trials=pickle.load(open(os.path.join(saved_dir,'hyperopt_trail_'+str(eps_now)+'.pkl'),"rb"))
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
                alpha=1,
                c='black')#cmap(float(i) / len(parameters)))
            axes[i].set_title(val)
            axes[i].set_ylim([np.array(ys).min()-0.1, np.array(ys).max()+0.1])
        plt.savefig(os.path.join(saved_dir,'hyperopt_trail_'+str(eps_now)+'.png'), bbox_inches='tight')
        trials_np=np.vstack(trials_list)
        np.save(os.path.join(saved_dir,'hyperopt_trail_np_'+str(eps_now)+'.npy'),trials_np)
        print(trials.best_trial)
        
        '''
        保存best table
        '''
        cal_table([best['t0'],best['t1'],best['t2']],saved_dir,clean_imgs,np.expand_dims(adv_imgs[idx_eps,...],axis=0),[eps_now])
    
    '''
    合并最终结果
    '''
    table_dict=dict()
    table_dict[0]=np.ones([8,8,3])
    for idx_eps,eps_now in enumerate(eps):
        table_pkl=os.path.join(saved_dir,'table_dict_'+str(eps_now)+'.pkl')
        tabel_dict_tmp=pickle.load(open(table_pkl,'rb'))
        table_dict.update(tabel_dict_tmp)
    pickle.dump(table_dict, open(os.path.join(saved_dir,'table_dict.pkl'),'wb'))
