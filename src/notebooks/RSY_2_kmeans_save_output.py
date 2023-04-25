#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import os
import sys
import importlib
import json
import uuid

import numpy as np
import pandas as pd
import scipy.ndimage as snd
import skimage
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
import plotly


# !py -m pip install numpy
# !py -m pip install pandas
# !py -m pip install scipy
# !py -m pip install opencv-python
# !py -m pip install matplotlib
# !py -m pip install plotly

if os.getcwd().split("/")[-1] == "notebooks":  # if cwd is located where this file is
    os.chdir("../..")  # go two folders upward (the if statement prevents error if cell is rerun)
directory_path = os.path.abspath(os.path.join("src"))
if directory_path not in sys.path:
    sys.path.append(directory_path)
print(directory_path)

import EyeTraumaAnalysis


# In[ ]:


importlib.reload(EyeTraumaAnalysis);


# # Unused testing functions

# In[10]:


### Per Cluster Masking ### <-- for individual image use
#image = EyeTraumaAnalysis.Image("data/01_raw/11000.jpg")
#draw_cluster_masking(image.img)
def draw_cluster_masking(img, K=10):
    row_ct = int(np.sqrt(K))
    col_ct = int(np.ceil(K/row_ct))
    fig, axs = plt.subplots(row_ct, col_ct, figsize=(12,6), sharex=True, sharey=True)
    for ind in range(row_ct*col_ct):
        if ind < K:
            #target1 = cv2.bitwise_and(image.img,image.img, mask=~kmeans_thresholds[ind])
            target1 = img.copy()
            target1[kmeans_thresholds[ind].astype(bool)] = [127,255,127,255]
            axs.flat[ind].imshow(target1)
            spatial_metrics = EyeTraumaAnalysis.get_spatial_metrics(kmeans_thresholds[ind])
            hsv_rank = centers_indices[ind]
            hsv_center = centers_sorted[ind]
            # Draw left title
            axs.flat[ind].set_title(
                "HSV \n"+
                f"#{hsv_rank[0]+1}, #{hsv_rank[1]+1}, #{hsv_rank[2]+1}" + "\n" +
                f"({hsv_center[0]}, {hsv_center[1]}, {hsv_center[2]})",
                fontsize=8, loc="left"
            )
            # Draw right title
            axs.flat[ind].set_title(
                f"Location:" + "\n"+
                f"({spatial_metrics['x']['mean']*100:.1f}, {spatial_metrics['y']['mean']:.1%})" + "\n" +
                f"({spatial_metrics['x']['sd']*100:.1f}, {spatial_metrics['y']['sd']:.1%})",
                fontsize=8, loc="right", fontfamily="monospace",
            )
            # axs.flat[ind].set_title(
            #     f"HSV center: [{centers_sorted[ind,0]},{centers_sorted[ind,1]},{centers_sorted[ind,2]}]" )
            #axs.flat[ind].imshow(kmeans_thresholds[ind], cmap="gray")
        else:
            # remove axes for empty cells
            axs.flat[ind].axis("off")


# In[11]:


#image = EyeTraumaAnalysis.Image("data/01_raw/11000.jpg")
#draw_separate_clusters(image.img, filename=None)
def draw_separate_clusters(img, filename=None, labelled=True, K_val=10):
    img_bgr = image.img
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Z_hsv = img_hsv.reshape((-1,3))
    # convert to np.float32
    Z_hsv = np.float32(Z_hsv)
    # calculate K-means segmentation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z_hsv,K_val,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    res_hsv = centers[label.flatten()]
    res_hsv2 = res_hsv.reshape(img_hsv.shape)
    res_bgr = cv2.cvtColor(res_hsv2, cv2.COLOR_HSV2BGR)
    centers_sorted = centers[centers[:, 2].argsort()]
    kmeans_thresholds = []
    for ind in range(K_val):
        kmeans_thresholds.append(cv2.inRange(res_hsv2,centers_sorted[ind],centers_sorted[ind]))
    centers_indices = centers_sorted.argsort(axis=0)
    for i in range(K_val):
        # propogate variables
        target = img.copy()
        # target[kmeans_thresholds[i].astype(bool)] = [127,255,127,255]
        target[np.where(kmeans_thresholds[i])] = [127,255,127,255]
        spatial_metrics = EyeTraumaAnalysis.get_spatial_metrics(kmeans_thresholds[i])
        hsv_rank, hsv_center = centers_indices[i], centers_sorted[i]
        # generate plot
        if labelled:
            plt.title(
                    "HSV \n"+
                    f"#{hsv_rank[0]+1}, #{hsv_rank[1]+1}, #{hsv_rank[2]+1}" + "\n" +
                    f"({hsv_center[0]}, {hsv_center[1]}, {hsv_center[2]})",
                    fontsize=8, loc="left"
                )
            plt.title(
                    f"Location:" + "\n"+
                    f"({spatial_metrics['x']['mean']*100:.1f}, {spatial_metrics['y']['mean']:.1%})" + "\n" +
                    f"({spatial_metrics['x']['sd']*100:.1f}, {spatial_metrics['y']['sd']:.1%})",
                    fontsize=8, loc="right", fontfamily="monospace",
                )
        plt.imshow(target)
        # save plot as PNG
        if filename is not None:
            if labelled:
                fpath = "C:/Users/ethan/PycharmProjects/EyeTraumaAnalysis/data/kmeans_indiv_clusters/labelled/"
            else:
                fpath = "C:/Users/ethan/PycharmProjects/EyeTraumaAnalysis/data/kmeans_indiv_clusters/non-labelled/"
            plt.savefig(f"{fpath}{filename}_v{i}.png", format="png")
        # plt.clf()


# #  Create clusters

# In[37]:


img_file_num = 14000
image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
img_hsv = cv2.cvtColor(np.float32(img_bgr)/255.0, cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
centers, ranges, res_hsv, kmeans_masks = EyeTraumaAnalysis.create_kmeans(img_hsv)
res_bgr = (cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR))
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)

plt.imshow(img_hsv)


# In[300]:


def create_kmeans(img, K=10, colorspace=None):  #
    """
    K is number of clusters
    colorspace doesn't change the actual arrays, just the columns names for the pandas dataframe outputted
    """
    channels = img.shape[-1]
    if colorspace is None:
        if channels==3:
            colorspace = "HSV"
        elif channels==4:
            colorspace = "BGRA"
        else:
            colorspace = "X" * channels

    img_linear = img.reshape((-1,channels))  # flatten shape part, but keep color dimension
    # NOTE: If you do cv2.cvtColor(.) to HSV on floats, the return HSV is from (0-360,0-1,0-1)
    # If you do it on uint8s (aka unsigned integers), they will be 0-255 just like the input
    # If you just do np.float32(.) or .astype(.) like below, then the original values are maintained
    # docs: https://docs.opencv.org/4.7.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
    img_linear = np.float32(img_linear)  # kmeans requires float32

    # Define criteria, arguments, and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    compactness, labels, centers = cv2.kmeans(img_linear,K,None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8 (if was uint8 originally), and make original image dimensions
    if img.dtype == "uint8":  # uint8 is 0-255 (unsigned 8 bit integer)
        centers = np.uint8(centers)
    res_img_flat = centers[labels.flatten()]   # shouldn't need to flatten as should already by x by 1
    res_img = res_img_flat.reshape(img.shape)
    labels = labels.reshape(img.shape[:2])

    # Sort centers by HSV "value" - aka sort by grayscale
    if colorspace.upper() in ["HSV"]:
        centers = centers[centers[:, 2].argsort()]
        #centers_indices = np.argsort(centers, axis=0)   # sorts each column separately
    elif colorspace.upper() in ["RGB","RGBA","BGR","BGRA"]:
        v = np.max(centers[:, :3], axis=1)  # the :3 is to remove an alpha channel if it exists
        centers = centers[v.argsort()]

    kmeans_masks = []
    for ind in range(K):
        # Can use opencv inRange or kmeans
        #kmeans_masks.append(cv2.inRange(res_img, centers[ind], centers[ind]))
        #kmeans_masks.append( np.all(res_img == centers[ind], axis=-1) )
        #kmeans_masks.append( res_img==centers[ind])
        # Below version works for floats as well
        kmeans_masks.append( labels == ind )
    kmeans_masks = np.array(kmeans_masks)

    # Couldn't make centers a DataFrame until now since needed numpy for opencv inRange or numpy comparison
    centers = pd.DataFrame(centers, columns=list(colorspace))   # list(.) converts "HSV" to ["H","S","V"]
    mins = pd.DataFrame([np.min(img[kmeans_mask],axis=0) for kmeans_mask in kmeans_masks], columns=list(colorspace))
    maxs = pd.DataFrame([np.min(img[kmeans_mask],axis=0) for kmeans_mask in kmeans_masks], columns=list(colorspace))
    clusters = pd.concat([centers,mins,maxs], axis=1, keys=["center","min","max"])
    clusters[("ct","#")] = np.sum(kmeans_masks, axis=(1,2))
    clusters[("ct","%")] = clusters[("ct","#")]/np.sum(clusters[("ct","#")])
    return centers, kmeans_masks, res_img, clusters


# In[228]:


def hsv_float32_to_uint8(img):
    """This converts from (0-360, 0-1, 0-1) range to (0-180, 0-255, 0-255) range
    """
    # NOTE: If you do cv2.cvtColor(.) to HSV on floats, the return HSV is from (0-360,0-1,0-1)
    # If you do it on uint8s (aka unsigned integers), they will be (0-180, 0-255, 0-255) just like the input
    # If you just do np.float32(.) or .astype(.) like below, then the original values are maintained
    # docs: https://docs.opencv.org/4.7.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
    img = img.copy()
    img[...,0] = img[...,0] *180/360   # H
    img[...,1] = img[...,1] *255   # S
    img[...,2] = img[...,2] *255   # V
    img = np.uint8(img)
    return img

def hsv_uint8_to_float32(img):
    """This converts from (0-180, 0-255, 0-255) range to (0-360, 0-1, 0-1) range
    """
    # NOTE: If you do cv2.cvtColor(.) to HSV on floats, the return HSV is from (0-360,0-1,0-1)
    # If you do it on uint8s (aka unsigned integers), they will be (0-180, 0-255, 0-255) just like the input
    # If you just do np.float32(.) or .astype(.) like below, then the original values are maintained
    # docs: https://docs.opencv.org/4.7.0/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
    img = np.float32(img.copy())
    img[...,0] = img[...,0] *360/180   # H
    img[...,1] = img[...,1] /255   # S
    img[...,2] = img[...,2] /255   # V
    return img


# # Run on HSV uint8

# In[311]:


img_file_num = 14000
image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
centers, kmeans_masks, res_img, clusters = create_kmeans(img_hsv)
res_bgr = np.uint8(cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR))
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)

plt.imshow(res_bgr)
with pd.option_context("display.precision", 3, "display.width",100):
    print(clusters)


# # Run on HSV float32

# In[306]:


img_file_num = 14000
image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
# HSV floats are from (0-360,0-1,0-1) while uint8s are 0-255 for all channels
img_hsv = cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
centers, kmeans_masks, res_img, clusters = create_kmeans(img_hsv)
res_bgr = np.uint8(cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR))
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
plt.imshow(res_bgr)
with pd.option_context("display.precision", 3, "display.width",100):
    print(clusters)


# # Run on BGR uint8

# In[312]:


img_file_num = 14000
image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
centers, kmeans_masks, res_img, clusters = create_kmeans(img_bgr)
clusters[("center","V")] = np.max(centers[["B","G","R"]], axis=1)
clusters = clusters.sort_values(by=("center","V"))
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
plt.imshow(res_bgr)
with pd.option_context("display.precision", 3, "display.width",100):
    print(clusters)


# # Run on old version of function (on hsv uint8)

# In[314]:


img_file_num = 14000
image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
# HSV floats are from (0-360,0-1,0-1) while uint8s are 0-255 for all channels
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
centers, ranges, res_hsv, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_hsv)
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
res_bgr = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)
plt.imshow(res_bgr)

centers["ct"] = np.sum(kmeans_masks, axis=(1,2))
centers["%"] = centers["ct"] / centers["ct"].sum()

print(centers)


# In[357]:


unique_vals = np.unique(masks_summed)
bounds = np.arange(0,255,25)-12.5

fig, axs = plt.subplots(2, 2, figsize=(12,6))
im0=axs.flat[0].imshow(img_bgr)
im1=axs.flat[1].imshow(res_bgr)
im2=axs.flat[2].imshow(masks_summed, cmap="gray")
im3=axs.flat[3].imshow(masks_summed, cmap="terrain")

cmap2 = mpl.cm.gray
norm2 = mpl.colors.BoundaryNorm(bounds, cmap2.N, extend='both')
cmap3 = mpl.cm.terrain
norm3 = mpl.colors.BoundaryNorm(bounds, cmap3.N, extend='both')


cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=axs.flat[2],
                     ticks=unique_vals, orientation="horizontal")
cbar3 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm3, cmap=cmap3), ax=axs.flat[3], cmap="terrain",
                     ticks=unique_vals, orientation="horizontal", shrink=0.9)
cbar2.ax.set_xticklabels([f"#{ind+1}" for ind in range(len(unique_vals))]);  # horizontal colorbar
cbar3.ax.set_xticklabels([f"#{ind+1}" for ind in range(len(unique_vals))]);  # horizontal colorbar

#plt.colorbar(im3,ax=axs.flat[3], shrink=0.8)


# In[15]:


img_file_num = 14000

for img_file_num in range(14000,14580):
    image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
    img_bgr = image.img
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    centers, ranges, res_hsv, kmeans_masks = EyeTraumaAnalysis.create_kmeans(img_hsv)
    masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
    plt.imsave(f"data/02_kmeans/{img_file_num}.png", masks_summed)


# In[ ]:


### Running Code on Several Images ###
from PIL import Image as PILimg

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
    # center indices
    centers_indices = centers_sorted.argsort(axis=0)
    ##
    summed_image = np.zeros(kmeans_thresholds[0].shape)
    for ind in range(K):
        summed_image += int(ind * 25) * kmeans_thresholds[ind] / 255
    plt.imsave(f"C:/Users/ethan/PycharmProjects/EyeTraumaAnalysis/data/all_clusters_indicated/{image_sample.split('.PNG')[0]}.png", summed_image)
    fig, axs = plt.subplots(2, 2, figsize=(12,6))
    im0=axs.flat[0].imshow(img_bgr)
    im1=axs.flat[1].imshow(res_bgr)
    im2=axs.flat[2].imshow(summed_image, cmap="gray")
    im3=axs.flat[3].imshow(summed_image, cmap="terrain")
    plt.colorbar(im3,ax=axs.flat[3], shrink=0.8)
    save_directory = f"{directory_path}".replace("src", "data")
    # plt.savefig(f"{save_directory}\\kmeans_clustering_applied\\ergonautus-10k\\clustered_view\\{image_sample.split('.PNG')[0]}.png",
    #             format='png')

    ## Per Cluster Masking
    row_ct = int(np.sqrt(K))
    col_ct = int(np.ceil(K/row_ct))
    fig, axs = plt.subplots(row_ct, col_ct, figsize=(12,6), sharex=True, sharey=True)
    for ind in range(row_ct*col_ct):
        if ind < K:
            #target1 = cv2.bitwise_and(image.img,image.img, mask=~kmeans_thresholds[ind])
            target1 = image.img.copy()
            target1[kmeans_thresholds[ind].astype(bool)] = [127,255,127,255]
            axs.flat[ind].imshow(target1)
            spatial_metrics = get_spatial_metrics(kmeans_thresholds[ind])
            hsv_rank = centers_indices[ind]
            hsv_center = centers_sorted[ind]
            # Draw left title
            axs.flat[ind].set_title(
                "HSV \n"+
                f"#{hsv_rank[0]+1}, #{hsv_rank[1]+1}, #{hsv_rank[2]+1}" + "\n" +
                f"({hsv_center[0]}, {hsv_center[1]}, {hsv_center[2]})",
                fontsize=8, loc="left"
            )
            # Draw right title
            axs.flat[ind].set_title(
                f"Location:" + "\n"+
                f"({spatial_metrics['x']['mean']*100:.1f}, {spatial_metrics['y']['mean']:.1%})" + "\n" +
                f"({spatial_metrics['x']['sd']*100:.1f}, {spatial_metrics['y']['sd']:.1%})",
                fontsize=8, loc="right", fontfamily="monospace",
            )
            # axs.flat[ind].set_title(
            #     f"HSV center: [{centers_sorted[ind,0]},{centers_sorted[ind,1]},{centers_sorted[ind,2]}]" )
            #axs.flat[ind].imshow(kmeans_thresholds[ind], cmap="gray")
        else:
            # remove axes for empty cells
            axs.flat[ind].axis("off")
        # save axis as singular image
        if (ind < 10):
            extent = axs.flat[ind].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # fig.savefig(f"C:/Users/ethan/PycharmProjects/EyeTraumaAnalysis/data/kmeans_indiv_clusters/{image_sample.split('.PNG')[0]}_v{ind}.png", format="png", bbox_inches = extent.expanded(1.1, 1.2))
    save_dir = "C:/Users/ethan/PycharmProjects/EyeTraumaAnalysis/data"
    # plt.savefig(f"{save_dir}\\kmeans_clustering_applied\\ergonautus-10k\\per_cluster_mask\\{image_sample.split('.PNG')[0]}.png", format='png')

    # row_ct = int(np.sqrt(K))
    # col_ct = int(np.ceil(K/row_ct))
    # fig, axs = plt.subplots(row_ct, col_ct, figsize=(12,6), sharex=True, sharey=True)
    # for ind in range(row_ct*col_ct):
    #     if ind < K:
    #         target1 = cv2.bitwise_and(image.img,image.img, mask=~kmeans_thresholds[ind])
    #         axs.flat[ind].imshow(target1)
    #         axs.flat[ind].set_title(
    #         f"Mean: {round(snd.mean(target1), 3)}\nStd: {round(snd.standard_deviation(target1), 3)}"
    #     )
    #         # axs.flat[ind].set_title(
    #         #     f"HSV center: [{centers_sorted[ind,0]},{centers_sorted[ind,1]},{centers_sorted[ind,2]}]" )
    #         #axs.flat[ind].imshow(kmeans_thresholds[ind], cmap="gray")
    #     else:
    #         # remove axes for empty cells
    #         axs.flat[ind].axis("off")
    # # plt.savefig(f"{save_directory}\\kmeans_clustering_applied\\K-{K}\\per_cluster_mask\\{image_sample.split('.jpg')[0]}.png", format='png')
    # plt.savefig(f"{save_directory}\\kmeans_clustering_applied\\ergonautus-10k\\per_cluster_mask\\{image_sample.split('.PNG')[0]}.png", format='png')

    # close to prevent overconsumption of memory
    plt.close()


# In[ ]:




