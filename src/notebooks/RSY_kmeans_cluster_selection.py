#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import importlib

import scipy.ndimage as snd

os.chdir("../..")
directory_path = os.path.abspath(os.path.join("src"))
if directory_path not in sys.path:
    sys.path.append(directory_path)

import EyeTraumaAnalysis

print(directory_path)
importlib.reload(EyeTraumaAnalysis);
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2


# In[6]:


folder = "./data/01_raw/Ergonautus/Full Dataset/"
image_filenames = os.listdir(folder)


# In[20]:


import shutil

for filename_old in image_filenames:
    #os.rename()
    [file_num_old, file_extension_old] = filename_old.split(".")
    file_num = int(file_num_old)
    # The original number system was messed up. It started from 000 and ended at 580 inclusive,
    # but skipped 205 (went directly from 204 to 206)
    if file_num > 205:
        file_num -= 1
    # Add 14000 to start with new first two digits, separating this dataset from the other datasets (e.g. diseased,
    # our own eyes, other healthy datasets from online)
    file_num += 14000
    filename_new = f"{file_num}.{file_extension_old.lower()}"
    file_path_old = os.path.join(folder, filename_old)
    file_path_new = os.path.join("./data/01_raw/", filename_new)
    shutil.copy(file_path_old, file_path_new)
    print(file_path_old, file_path_new)

