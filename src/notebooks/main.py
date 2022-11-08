#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import importlib
os.chdir("../..")
directory_path = os.path.abspath(os.path.join("src"))
if directory_path not in sys.path:
    sys.path.append(directory_path)

import EyeTraumaAnalysis


# In[4]:


importlib.reload(EyeTraumaAnalysis)


# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[16]:


image = EyeTraumaAnalysis.Image("data/ischemic/00001_li.jpg")
plt.imshow(image.img)


# In[14]:


segments = EyeTraumaAnalysis.get_segments(
    img=image.img,
    interval_deg=10,
    wd_px=20,
    center=image.center)


# In[7]:


plt.imshow(segments[0])


# In[8]:


plt.imshow(np.vstack(segments))


# In[ ]:




