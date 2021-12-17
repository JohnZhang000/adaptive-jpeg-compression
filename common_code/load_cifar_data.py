#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:51:04 2021

@author: ubuntu204
"""
import os 
import numpy as np
from six.moves import cPickle as pickle
# from scipy.misc import imread
import platform
from PIL import Image

def load_pickle(f):
    version = platform.python_version_tuple() # 取python版本号
    if version[0] == '2':
        return  pickle.load(f) # pickle.load, 反序列化为python的数据类型
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)   # dict类型
    X = datadict['data']        # X, ndarray, 像素值
    Y = datadict['labels']      # Y, list, 标签, 分类
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)/255.0
    Y = np.array(Y)
    return X, Y

def load_CIFAR_train(filename):
  """ load single batch of cifar """
  data_list = []
  label_list = []
  for i in range(1,6):
      file = 'data_batch_{0}'.format(i)
      f = os.path.join(filename,file)
      data, label = load_CIFAR_batch(f)
      data_list.append(data)
      label_list.append(label)
  X = np.concatenate(data_list)
  Y = np.concatenate(label_list)
  return X,Y

def load_imagenet_filenames(dataset_dir,features):
    filename=dataset_dir+'.txt'
    with open(filename, 'r') as f:
        data_list=f.readlines()
        
    label_list=[]
    image_list=[]
    for data in  data_list:
        sysnet,name=data.split('/')
        label_list.append(features[sysnet])
        image_list.append(data.replace('\n',''))
    return image_list,label_list
    
def load_imagenet_batch(batch_idx,batch_size,data_dir,data_list,label_list):
    filenames=data_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
    labels=np.array(label_list[batch_idx*batch_size:(batch_idx+1)*batch_size])
    
    images=np.zeros([batch_size,224,224,3])
    for file_idx,file in enumerate(filenames):
        image = Image.open(os.path.join(data_dir,file)).convert('RGB').resize([224,224])
        images[file_idx,...] = np.asarray(image)/255.0
    images=images.transpose(0,3,1,2).astype(np.float32)
    # images=images
    return images,labels