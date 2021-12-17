#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 17:33:03 2021

@author: ubuntu204
"""
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as multitest
import matplotlib.pyplot as plt
import os

import pandas as pd
from pandas import Series,DataFrame
# import seaborn as sns
# import palettable
from sklearn import datasets
from tqdm import tqdm

plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_mnius']=False
epsilon=1e-10

def volcano_mine(data1,data2,method='hs',flag_output_src=0,flag_plot=0):
    data1=data1+epsilon
    data2=data2+epsilon
    
    mdata1=data1.mean(axis=0)
    mdata2=data2.mean(axis=0)    
    fold_change=(mdata2)/(mdata1)
    log2_fold_change=np.log2(fold_change)
    
    p_values=np.zeros_like(mdata1)
    for i in tqdm(range(len(p_values))):
        t,p=stats.ttest_ind(data1[:,i],data2[:,i])
        p_values[i]=p
    rejects,pvals_corrected,alphaSidak,alphaBonf=multitest.multipletests(p_values,method=method)
    log10_pvals_corrected=np.log10(pvals_corrected+epsilon)*(-1)
    
    return log2_fold_change,log10_pvals_corrected
    
def plot_volume(log2_fold_change,log10_pvals_corrected,title=None,saved_name=None):
    npt=len(log2_fold_change)
    colors=list(['grey']*npt)
    idx_green=(log2_fold_change>=np.log2(1.2))&(log10_pvals_corrected>(-np.log10(0.05)))
    for i in range(len(idx_green)):
        if idx_green[i]:
            colors[i]='green'
    idx_red=(log2_fold_change<=-np.log2(1.2))&(log10_pvals_corrected>(-np.log10(0.05)))
    for i in range(len(idx_red)):
        if idx_red[i]:
            colors[i]='red'
    # colors[idx_red]='red'
    
    plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.scatter(log2_fold_change, log10_pvals_corrected, color=colors)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 P-Value')
    if title:
        plt.title(title)
    if saved_name:
        plt.savefig(saved_name,bbox_inches='tight',dpi=300)
    return

# def plot_heatmap(data,row_c=None,dpi=300,figsize=(8/2.54,16/2.54),saved_name=None):
#     # plt.figure(dpi=dpi)
#     data_show=data.copy()
#     # data_show=data.drop(['class'],axis=1)
#     if row_c:
#         row_colors=data['class'].map(row_c)
#     sns.clustermap(data=data_show,method='single',metric='euclidean',
#                    figsize=figsize,row_cluster=False,col_cluster=False,
#                    cmap='rainbow')
#     sns.set(font_scale=1.5)
#     if saved_name:
#         plt.savefig(saved_name,bbox_inches='tight',dpi=dpi)
    
    
if __name__=='__main__':    
    # data1=np.random.rand(5, 10)
    # data2=np.random.rand(5, 10)
    # data2[:,0]=data1[:,0]*2.5
    # data2[:,1]=data1[:,1]*10
    # data2[:,2]=data1[:,2]/2.5
    # data2[:,3]=data1[:,3]/10
    # logFC,logP=volcano_mine(data1, data2)
    # plot_volume(logFC,logP)
    
    iris=datasets.load_iris()
    x,y=iris.data,iris.target
    data=np.hstack((x,y.reshape(150,1)))
    pd_iris=pd.DataFrame(data,columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'])
    row_c=dict(zip(pd_iris['class'].unique(),['green','yellow','pink']))
    # plot_heatmap(pd_iris,row_c=row_c)