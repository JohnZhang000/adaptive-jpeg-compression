#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:48:27 2021

@author: dell
"""
import matplotlib as mpl
# mpl.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

class img_ploter:
    
    # 解释器初始化
    def __init__(self,font='Times New Roman',fontsize=8,saved_width=8):
        a=1
        self.colors=['r','g','k','c','m','b','w']
        # marks=['d','D','x','+','H','h','*']
        self.marks = ['-o', '-^', '-3', '-P', '-x', '-_', '-v', '-2', '-p', '-+', '-1', '-s', '-H', '-d', '-,', '->', '-8', '-h', '-D', '-.', '-<', '-4', '-*', '-X']
        self.img_format='.jpg'
        # self.fontsize=8
        self.font = font
        self.fontsize=fontsize
        self.fontprop={'family':self.font,'weight':'normal','size':self.fontsize}
        self.fontprop_legend={'family':self.font,'weight':'normal','size':self.fontsize}
        self.cm2inch=1/2.54
        self.saved_dpi=350
        self.saved_width=saved_width
        self.colors1=[(255/255,59/255,59/255)]
        self.colors2=[(1/255,86/255,153/255),(250/255,192/255,15/255)]
        self.colors3=[(1/255,86/255,153/255),(250/255,192/255,15/255),(243/255,118/255,74/255)]
        self.colors4=[(1/255,86/255,153/255),(250/255,192/255,15/255),(243/255,118/255,74/255),(95/255,198/255,201/255)]
        self.colors5=[(1/255,86/255,153/255),(250/255,192/255,15/255),(243/255,118/255,74/255),(95/255,198/255,201/255),(79/255,89/255,109/255)]
        self.markersize=2
        self.linewidth=1
        
        plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
        plt.rc('font',family='Times New Roman')

    def plot_line(self,x,data,data_group,xlabel=None,ylabel=None,tick_label=None,xtick=None,title=None,saved_name=None,saved_width=None,height_scale=0.618,colors_in=None,xlim=None,ylim=None):
        if colors_in:
            colors=colors_in
        else:
            colors=eval('self.colors'+str(len(data_group)))
        marks=self.marks
        method_idx=0
        if not saved_width:
            saved_width=self.saved_width
        plt.figure(figsize=(saved_width*self.cm2inch,height_scale*saved_width*self.cm2inch), dpi=self.saved_dpi)
        for i in range(len(data_group)):         
            for j in range(data_group[i]):
                y=data[method_idx,:]
                if len(data_group)<6:
                    if tick_label:
                        plt.plot(x, y, marks[j],label=tick_label[method_idx],color=colors[i],markersize=self.markersize,linewidth=self.linewidth)
                    else:
                        plt.plot(x, y, marks[j],color=colors[i],markersize=self.markersize,linewidth=self.linewidth)
                
                method_idx+=1       
        if not xtick:
            xtick=[str(i) for i in x]
        plt.xticks(x, xtick)
        if xlabel:
            plt.xlabel(xlabel,fontproperties =self.font, size = self.fontsize)
        if ylabel:
            plt.ylabel(ylabel,fontproperties =self.font, size = self.fontsize)
        if tick_label:
            # plt.legend(tick_label,bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0,prop=self.fontprop) 
            plt.legend(loc=0, borderaxespad=0,prop=self.fontprop_legend) 
        if title:
            plt.title(title)
        if xlim:
            plt.xlim(xlim) 
        if ylim:
            plt.ylim(ylim) 
        plt.grid()
        plt.yticks(fontproperties =self.font, size = self.fontsize)
        plt.xticks(fontproperties =self.font, size = self.fontsize)
        # plt.gcf().subplots_adjust(left=0.15,right=0.68,bottom=0.15)
           
        plt.tight_layout()
        if saved_name:
            print(saved_name+self.img_format)
            # plt.savefig('1.jpg')
            plt.savefig(saved_name+self.img_format, dpi=self.saved_dpi)
        
    def plot_bar(self,y,ylabel=None,tick_label=None,title=None,saved_name=None,saved_width=None,height_scale=0.618):
        x=np.arange(len(y))
        if not saved_width:
            saved_width=self.saved_width
        plt.figure(figsize=(saved_width*self.cm2inch,height_scale*saved_width*self.cm2inch), dpi=self.saved_dpi)
        # plt.grid()
        plt.bar(x, height=y, width=0.5)
        y_diff=max(y)-min(y)
        for a, b in zip(x, y):            
            plt.text(a, b + 0.05*y_diff, '%.2f' % b, ha='center', va='bottom',family=self.font,size=self.fontsize)
        y_max=max(y)+0.2*y_diff
        y_min=max(0,min(y)-0.2*y_diff)
        plt.ylim((y_min,y_max))
        plt.yticks(fontproperties =self.font, size = self.fontsize)
        if ylabel:
            plt.ylabel(ylabel,fontproperties =self.font, size = self.fontsize)
        if tick_label:
            plt.xticks(x,tick_label,rotation=30,fontproperties =self.font, size = self.fontsize)
        if title:
            plt.title(title)
        plt.tight_layout()
        if saved_name:
            plt.savefig(saved_name+self.img_format, dpi=self.saved_dpi)
        
        
    def plt_line_conf(self,data_in,flag_fill=1,xlim=None,ylim=None,xlabel=None,ylabel=None,tick_label=None,title=None,saved_name=None,saved_width=None,height_scale=0.618,colors_in=None):
        if not saved_width:
            saved_width=self.saved_width
        plt.figure(figsize=(saved_width*self.cm2inch,height_scale*saved_width*self.cm2inch), dpi=self.saved_dpi)
        data_num=len(data_in)
        if colors_in:
            colors=colors_in
        else:
            if data_num<6:
                colors=eval('self.colors'+str(data_num))
            else:
                colors=None
        for i in range(data_num):
            y=data_in[i]
            y_mean = y.mean(axis=0)
        #    y_mean = np.median(data_in,axis=0)
            y_std  = y.std(axis=0)
            x      = range(len(y_mean))
            if data_num<6:
                plt.plot(x, y_mean, color=colors[i],linewidth=self.linewidth)
                if flag_fill:
                    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, facecolor=colors[i])
            else:
                plt.plot(x, y_mean, linewidth=self.linewidth)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        if xlabel:
            plt.xlabel(xlabel,fontproperties =self.font, size = self.fontsize)
        if ylabel:
            plt.ylabel(ylabel,fontproperties =self.font, size = self.fontsize)
        if tick_label:
            plt.legend(tick_label,loc=0, borderaxespad=0,prop=self.fontprop_legend)       
        if title:
            plt.title(title)
        plt.grid()
        plt.yticks(fontproperties =self.font, size = self.fontsize)
        plt.xticks(fontproperties =self.font, size = self.fontsize)
        plt.tight_layout()
        if saved_name:
            plt.savefig(saved_name+self.img_format, dpi=self.saved_dpi)
            
    def plt_line_conf_x(self,data_in,x=None,flag_fill=0,xlabel=None,ylabel=None,tick_label=None,title=None,saved_name=None,saved_width=None,height_scale=0.618,colors_in=None):
        if not saved_width:
            saved_width=self.saved_width
        plt.figure(figsize=(saved_width*self.cm2inch,height_scale*saved_width*self.cm2inch), dpi=self.saved_dpi)
        data_num=len(data_in)
        if colors_in:
            colors=colors_in
        else:
            if data_num<6:
                colors=eval('self.colors'+str(data_num))
            else:
                colors=None
        for i in range(data_num):
            y=data_in[i]
            y_mean = y.mean(axis=0)
        #    y_mean = np.median(data_in,axis=0)
            y_std  = y.std(axis=0)
            if x is None:
                x      = range(len(y_mean))
            if data_num<6:
                plt.plot(x, y_mean, color=colors[i],linewidth=self.linewidth)
                if flag_fill:
                    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, facecolor=colors[i])
            else:
                plt.plot(x, y_mean, linewidth=self.linewidth)

        if xlabel:
            plt.xlabel(xlabel,fontproperties =self.font, size = self.fontsize)
        if ylabel:
            plt.ylabel(ylabel,fontproperties =self.font, size = self.fontsize)
        if tick_label:
            plt.legend(tick_label,loc=0, borderaxespad=0,prop=self.fontprop_legend)       
        if title:
            plt.title(title)
        plt.grid()
        plt.yticks(fontproperties =self.font, size = self.fontsize)
        plt.xticks(fontproperties =self.font, size = self.fontsize)
        plt.tight_layout()
        if saved_name:
            plt.savefig(saved_name+self.img_format, dpi=self.saved_dpi)


        
        
if __name__=='__main__':  
    ploter=img_ploter()
    
    # y=np.random.randn(5)*10
    y=np.random.randint(0, 100, 5)
    tick_label=['M1','M2','M3','M4','M5']
    ploter.plot_bar(y,ylabel='Accuracy (%)',tick_label=tick_label,saved_name='bar')
    
    y1=np.random.randn(5,5)
    y2=np.random.randn(5,5)
    y3=np.random.randn(5,5)
    y4=np.random.randn(5,5)
    tick_label=['L0S20','L1S20','M3','L4']
    ploter.plt_line_conf([y1,y2,y3,y4],ylabel='Confidence',tick_label=tick_label,saved_name='line_conf')
    
    y1=np.random.randn(14,5)
    tick_label=['L0S20','L0S10','L0S5','L1S10','L2S10','L3S10','e0.01','e0.05','e0.1','SIN','SIN_IN','SIN_INF','AUG_MIX','vanilla']
    # x=[0.01,0.02,0.03,0.04,0.05]
    x=[1,2,3,4,5]
    data_group=[6,3,4,1]
    colors=None#[(210/255,32/255,39/255), (28/255,28/255,28/255),  (105/255,105/255,105/255),  (207/255,207/255,207/255)]
    ploter.plot_line(x,y1,data_group,tick_label=tick_label,saved_width=8,height_scale=0.9,saved_name='line',colors_in=colors)
