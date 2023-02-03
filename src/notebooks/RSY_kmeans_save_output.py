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


# In[2]:


print(directory_path)


# In[2]:


importlib.reload(EyeTraumaAnalysis);


# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2


# In[5]:


# images = ["10030.jpg", "10031.jpg", "10032.jpg", "10033.jpg", "10034.jpg", "10035.jpg", "10036.jpg", "10037.jpg", "10038.jpg", "10039.jpg", "10040.jpg", "10041.jpg", "10042.jpg"]


# In[4]:


images = os.listdir('./data/01_raw/Ergonautus/Full Dataset/')


# In[6]:


### Create K means clusters and masks
image = EyeTraumaAnalysis.Image("data/01_raw/11000.jpg")
img_bgr = image.img
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
Z_hsv = img_hsv.reshape((-1,3))
# convert to np.float32
Z_hsv = np.float32(Z_hsv)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
ret,label,centers=cv2.kmeans(Z_hsv,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
centers = np.uint8(centers)
res_hsv = centers[label.flatten()]
res_hsv2 = res_hsv.reshape(img_hsv.shape)
res_bgr = cv2.cvtColor(res_hsv2, cv2.COLOR_HSV2BGR)
# res2 = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)


# sort centers by HSV "value" - aka sort by grayscale
centers_sorted = centers[centers[:, 2].argsort()]
kmeans_thresholds = []
for ind in range(K):
    kmeans_thresholds.append(cv2.inRange(res_hsv2,centers_sorted[ind],centers_sorted[ind]))
summed_image = np.zeros(kmeans_thresholds[0].shape)
for ind in range(K):
    summed_image += int(ind/K*255) * kmeans_thresholds[ind]


# In[7]:


### Clustered View ### <-- for individual image use
fig, axs = plt.subplots(2, 2, figsize=(12,6))
im0=axs.flat[0].imshow(img_bgr)
im1=axs.flat[1].imshow(res_bgr)
im2=axs.flat[2].imshow(summed_image, cmap="gray")
im3=axs.flat[3].imshow(summed_image, cmap="terrain")
plt.colorbar(im3,ax=axs.flat[3], shrink=0.8)


# In[53]:


def get_spatial_metrics(mask):
    # scipy can perform the mean (center of mass), but not the standard deviation
    # spatial_means = snd.center_of_mass(mask)
    x = np.arange(0, mask.shape[1])
    y = np.arange(0, mask.shape[0])
    xgrid, ygrid = np.meshgrid(x, y)
    grids = {"x": xgrid, "y":ygrid}
    to_return = {"x":{}, "y":{}}
    for ind, grid in grids.items():
        to_return[ind]["mean"] = np.mean(grids[ind], where=mask.astype(bool))
        to_return[ind]["sd"] = np.std(grids[ind], where=mask.astype(bool))
    return to_return
    pass


# In[8]:


### Per Cluster Masking ### <-- for individual image use

row_ct = int(np.sqrt(K))
col_ct = int(np.ceil(K/row_ct))
fig, axs = plt.subplots(row_ct, col_ct, figsize=(12,6), sharex=True, sharey=True)
for ind in range(row_ct*col_ct):
    if ind < K:
        target1 = cv2.bitwise_and(image.img,image.img, mask=~kmeans_thresholds[ind])
        axs.flat[ind].imshow(target1)

        spatial_mean = snd.center_of_mass(kmeans_thresholds[ind])
        x = np.arange(0, target1.shape[1])
        y = np.arange(0, target1.shape[0])
        xgrid, ygrid = np.meshgrid(x, y)
        xgrid = xgrid.flatten()[kmeans_thresholds[ind].flatten()]
        ygrid = ygrid.flatten()[kmeans_thresholds[ind].flatten()]
        axs.flat[ind].set_title(
            f"Mean: {spatial_mean[1]:.2f}, {spatial_mean[0]:.2f}\n" +
            f"SD: {snd.mean(xgrid):.2f}, {snd.mean(ygrid):.2f}"
        )
        # axs.flat[ind].set_title(
        #     f"HSV center: [{centers_sorted[ind,0]},{centers_sorted[ind,1]},{centers_sorted[ind,2]}]" )
        #axs.flat[ind].imshow(kmeans_thresholds[ind], cmap="gray")
    else:
        # remove axes for empty cells
        axs.flat[ind].axis("off")


# In[23]:


kmeans_thresholds[0].dtype


# In[27]:


xgrid


# In[22]:


type(xgrid)


# In[ ]:


### Running Code on Several Images ###

K = 10
# save_directory = "C:\\Users\\ethan\\PycharmProjects\\EyeTraumaAnalysis\\data\\kmeans_clustering_applied" # data/kmeans_clustering_applied; hard coded for PC
# NOTE: While this isn't a preferable implementation, the previous code, which was flexible per system, ran into PermissionError [Errno 13]

for image_sample in images:
    # image = EyeTraumaAnalysis.Image(f"data/01_raw/{image_sample}")
    image = EyeTraumaAnalysis.Image(f"data/01_raw/Ergonautus/Full Dataset/{image_sample}")

    ## Clustered View
    img_bgr = image.img
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Z_hsv = img_hsv.reshape((-1,3))
    Z_hsv = np.float32(Z_hsv)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z_hsv,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res_hsv = centers[label.flatten()]
    res_hsv2 = res_hsv.reshape(img_hsv.shape)
    res_bgr = cv2.cvtColor(res_hsv2, cv2.COLOR_HSV2BGR)
    centers_sorted = centers[centers[:, 2].argsort()]
    kmeans_thresholds = []
    for ind in range(K):
        kmeans_thresholds.append(cv2.inRange(res_hsv2,centers_sorted[ind],centers_sorted[ind]))
    summed_image = np.zeros(kmeans_thresholds[0].shape)
    for ind in range(K):
        summed_image += int(ind/K*255) * kmeans_thresholds[ind]
    fig, axs = plt.subplots(2, 2, figsize=(12,6))
    im0=axs.flat[0].imshow(img_bgr)
    im1=axs.flat[1].imshow(res_bgr)
    im2=axs.flat[2].imshow(summed_image, cmap="gray")
    im3=axs.flat[3].imshow(summed_image, cmap="terrain")
    plt.colorbar(im3,ax=axs.flat[3], shrink=0.8)
    save_directory = f"{directory_path}".replace("src", "data")
    # plt.savefig(f"{save_directory}\\kmeans_clustering_applied\\K-{K}\\clustered_view\\{image_sample.split('.PNG')[0]}.png",
    #             format='png')
    plt.savefig(f"{save_directory}\\kmeans_clustering_applied\\ergonautus-10k\\clustered_view\\{image_sample.split('.PNG')[0]}.png",
                format='png')

    ## Per Cluster Masking
    row_ct = int(np.sqrt(K))
    col_ct = int(np.ceil(K/row_ct))
    fig, axs = plt.subplots(row_ct, col_ct, figsize=(12,6), sharex=True, sharey=True)
    for ind in range(row_ct*col_ct):
        if ind < K:
            target1 = cv2.bitwise_and(image.img,image.img, mask=~kmeans_thresholds[ind])
            axs.flat[ind].imshow(target1)
            axs.flat[ind].set_title(
            f"Mean: {round(snd.mean(target1), 3)}\nStd: {round(snd.standard_deviation(target1), 3)}"
        )
            # axs.flat[ind].set_title(
            #     f"HSV center: [{centers_sorted[ind,0]},{centers_sorted[ind,1]},{centers_sorted[ind,2]}]" )
            #axs.flat[ind].imshow(kmeans_thresholds[ind], cmap="gray")
        else:
            # remove axes for empty cells
            axs.flat[ind].axis("off")
    # plt.savefig(f"{save_directory}\\kmeans_clustering_applied\\K-{K}\\per_cluster_mask\\{image_sample.split('.jpg')[0]}.png", format='png')
    plt.savefig(f"{save_directory}\\kmeans_clustering_applied\\ergonautus-10k\\per_cluster_mask\\{image_sample.split('.PNG')[0]}.png", format='png')

    # close to prevent overconsumption of memory
    plt.close()

