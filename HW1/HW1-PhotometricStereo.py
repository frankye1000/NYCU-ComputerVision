#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, norm
from sklearn.preprocessing import normalize

from HW1 import *


# # 圖片來源種類

# In[2]:


source='star'


# # read LightSource.txt
# 讀取光源

# In[3]:


with open("test/{}/LightSource.txt".format(source),'r') as f:
    L = []
    for l in f.readlines():
        l = l.replace('\n','').replace('(','').replace(')','').split(' ')[1].split(',')
        l = [int(i) for i in l]
        L.append(l)

# to array        
L = np.array(L)


# In[4]:


L


# # read images
# 讀取影像

# In[5]:


img1 = read_bmp('test/{}/pic1.bmp'.format(source)).reshape(-1)
img2 = read_bmp('test/{}/pic2.bmp'.format(source)).reshape(-1)
img3 = read_bmp('test/{}/pic3.bmp'.format(source)).reshape(-1)
img4 = read_bmp('test/{}/pic4.bmp'.format(source)).reshape(-1)
img5 = read_bmp('test/{}/pic5.bmp'.format(source)).reshape(-1)
img6 = read_bmp('test/{}/pic6.bmp'.format(source)).reshape(-1)


# In[6]:


I = np.vstack((img1,img2,img3,img4,img5,img6))
I.shape


# # KdN
# 利用pesudo-inverse計算KdN

# In[7]:


KdN = pinv((L.T@L))@L.T@I
KdN = KdN.T  # 需要再轉置一次，畫出來結果才是對的


# In[24]:


# 正規化
N = normalize(KdN, axis=1, norm="l2")


# # 影像呈現

# In[25]:


normal_visualization(N)


# In[ ]:




