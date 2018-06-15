# -*- coding: utf-8 -*-
"""
Spyder Editor

KUMASTORM
"""
import scipy.io as scio

import pandas as pd
#mat文件位置
data_path="images/VehicleInfo.mat"

data = scio.loadmat(data_path,squeeze_me=1)
#选取1920*1280分辨率照片的前500张作为训练数据。
data_v = data['VehicleInfo'][:500]

#初始化参数表
v_labels = pd.DataFrame()
v_labels_f = []
v_labels_w = []
v_labels_h = []
v_labels_c = []
v_labels_xmin = []
v_labels_ymin = []
v_labels_xmax = []
v_labels_ymax = []

for x in data_v:
    for y in x[3].reshape(-1):
        v_labels_f.append(x[0])
        v_labels_w.append(x[1])
        v_labels_h.append(x[2])
        v_labels_xmin.append(y[0])
        v_labels_ymin.append(y[1])
        v_labels_xmax.append(y[2])
        v_labels_ymax.append(y[3])
#        这里为了便于训练，统一把类别替换成了vehicle，实际中如果需要识别各类车辆，
#        直接使用y[4]即可读入车辆分类。
        v_labels_c.append('vehicle')
#        v_labels_c.append(y[4])
            
v_labels['filename']	= v_labels_f
v_labels['width']	= v_labels_w
v_labels['height']	= v_labels_h
v_labels['class']	= v_labels_c
v_labels['xmin']	= v_labels_xmin
v_labels['ymin']	= v_labels_ymin
v_labels['xmax']	= v_labels_xmax
v_labels['ymax']	= v_labels_ymax

#转为所需CSV格式，这里一共采用了500张图片，改为400张训练，100张检验。由于部分文件里面
#不止一辆车，所以实际index为424
v_labels.iloc[:425,:].to_csv('train_data/v_labels_train.csv',index = False)
v_labels.iloc[425:,:].to_csv('train_data/v_labels_test.csv',index = False)