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


# # Test how cv2.cvtColor(.) works when input is uint8 vs float32
# Note: float64 is not supported by opencv kmeans so preemptively using float32 only

# In[5]:


img_file_num = 14000
image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img


# In[5]:


a1=cv2.cvtColor(np.float32(img_bgr)*2, cv2.COLOR_BGR2HSV)
b1=cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2HSV)

a2 = cv2.cvtColor(a1, cv2.COLOR_HSV2BGR)
b2 = cv2.cvtColor(b1, cv2.COLOR_HSV2BGR)

a1m,b1m=np.max(a1[...,2]), np.max(b1[...,2])
a2m,b2m=np.max(a2[...,2]), np.max(b2[...,2])
(a1m,b1m,a1m/b1m), (a2m,b2m,a2m/b2m)


# In[6]:


a1=cv2.cvtColor(np.uint8(img_bgr), cv2.COLOR_BGR2HSV)
b1=cv2.cvtColor(np.float32(img_bgr)/255, cv2.COLOR_BGR2HSV)

a2 = cv2.cvtColor(a1, cv2.COLOR_HSV2BGR)
b2 = cv2.cvtColor(b1, cv2.COLOR_HSV2BGR)*255

a1m,b1m=np.max(a1[...,2]), np.max(b1[...,2])
a2m,b2m=np.max(a2[...,2]), np.max(b2[...,2])
(a1m,b1m,a1m/b1m), (a2m,b2m,a2m/b2m)


# In[7]:


a1=cv2.cvtColor(np.uint8(img_bgr), cv2.COLOR_BGR2HSV)
b1=cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2HSV)

a2 = cv2.cvtColor(a1, cv2.COLOR_HSV2BGR)
b2 = cv2.cvtColor(b1, cv2.COLOR_HSV2BGR)

a1m,b1m=np.mean(a1[...,2]), np.mean(b1[...,2])
a2m,b2m=np.mean(a2[...,2]), np.mean(b2[...,2])
(a1m,b1m,a1m/b1m), (a2m,b2m,a2m/b2m)


# In[12]:


a1=cv2.cvtColor(np.uint8(img_bgr), cv2.COLOR_BGR2HSV)
b1=cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2HSV)

a2 = cv2.cvtColor(a1, cv2.COLOR_HSV2BGR)
b2 = cv2.cvtColor(b1, cv2.COLOR_HSV2BGR)

a1m,b1m=np.max(a1, axis=(0,1)), np.max(b1, axis=(0,1))
a2m,b2m=np.max(a2, axis=(0,1)), np.max(b2, axis=(0,1))
(a1m,b1m,a1m/b1m), (a2m,b2m,a2m/b2m)


# # Display subplot analysis

# In[73]:


def plot_kmeans_subplots_view(img_bgr, res_bgr, masks_summed):
    unique_vals = np.unique(masks_summed)
    bounds = np.arange(0,255,25)-12.5

    cmap2 = mpl.cm.gray
    norm2 = mpl.colors.BoundaryNorm(bounds, cmap2.N, extend='both')
    cmap3 = mpl.cm.terrain
    norm3 = mpl.colors.BoundaryNorm(bounds, cmap3.N, extend='both')

    fig, axs = plt.subplots(2, 2, figsize=(12,6))
    im0=axs.flat[0].imshow(img_bgr)
    im1=axs.flat[1].imshow(res_bgr)
    im2=axs.flat[2].imshow(masks_summed, cmap=cmap2)
    im3=axs.flat[3].imshow(masks_summed, cmap=cmap3)
    for ind in range(len(axs.flat)):
        axs.flat[ind].axis("off")  # takes off x and y axes ticks

    cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=axs.flat[2],
                         ticks=unique_vals, orientation="horizontal")
    cbar3 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm3, cmap=cmap3), ax=axs.flat[3], cmap="terrain",
                         ticks=unique_vals, orientation="horizontal", shrink=0.9)
    cbar2.ax.set_xticklabels([f"#{ind+1}" for ind in range(len(unique_vals))]);  # horizontal colorbar
    cbar3.ax.set_xticklabels([f"#{ind+1}" for ind in range(len(unique_vals))]);  # horizontal colorbar

    plt.tight_layout()

def plot_kmeans_subplots_view_simple(img_bgr, res_bgr, masks_summed):
    unique_vals = np.unique(masks_summed)
    bounds = np.arange(0,255,25)-12.5

    cmap2 = mpl.cm.terrain
    norm2 = mpl.colors.BoundaryNorm(bounds, cmap2.N, extend='both')

    fig, axs = plt.subplots(1, 3, figsize=(8,2))   # 1 row, 3 columns
    im0=axs.flat[0].imshow(img_bgr)
    im1=axs.flat[1].imshow(res_bgr)
    im2=axs.flat[2].imshow(masks_summed, cmap=cmap2)
    for ind in range(len(axs.flat)):
        axs.flat[ind].axis("off")  # takes off x and y axes ticks


    cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2), ax=axs.flat[2],
                         ticks=unique_vals, orientation="horizontal")
    cbar2.ax.set_xticklabels([f"#{ind+1}" for ind in range(len(unique_vals))]);  # horizontal colorbar
    plt.tight_layout()

def plot_per_cluster_masking(img_bgr, centers, kmeans_masks, src_info={},
                             color=[127,255,127,255],  # bright green
                             ):
    ch = img_bgr.shape[2]
    if color is None:
        if ch==4:
            color = [127,255,127,255]  # bright green
        elif ch==3:
            color = [127,255,127]      # bright green
        else:
            color = [127] * ch   # gray

    if src_info.get("file_num") is not None:
        img_file_num = src_info.get("file_num")
        img_race_abbr, img_race_long = EyeTraumaAnalysis.ergonautas_file_num_to_race(img_file_num)

    K = kmeans_masks.shape[0]
    centers_indices = np.argsort(centers, axis=0)  # get ranks

    row_ct = int(np.sqrt(K))
    col_ct = int(np.ceil(K/row_ct))
    fig, axs = plt.subplots(row_ct, col_ct, figsize=(12,6), sharex=True, sharey=True)
    for ind in range(row_ct*col_ct):
        if ind < K:
            target1 = img_bgr.copy()   # the background is the original image
            target1[kmeans_masks[ind].astype(bool)] = color
            axs.flat[ind].imshow(target1)

            spatial_metrics = EyeTraumaAnalysis.get_spatial_metrics(kmeans_masks[ind])
            hsv_center = centers.iloc[ind]
            hsv_rank = centers_indices.iloc[ind]
            area_pct = np.sum(kmeans_masks[ind]) / np.product(kmeans_masks[ind].shape)

            # Draw left title
            axs.flat[ind].set_title(
                "   COLOR:" + "\n" +
                " H ,  S ,  V " + "\n" +
                f"#{hsv_rank[0]+1:<2.0f}, #{hsv_rank[1]+1:<2.0f}, #{hsv_rank[2]+1:<2.0f}" + "\n" +
                f"{hsv_center[0]:^3.0f}, {hsv_center[1]:^3.0f}, {hsv_center[2]:^3.0f}",
                fontsize=8, loc="left", fontfamily="monospace",
            )
            # Draw right title
            axs.flat[ind].set_title(
                f"LOCATION:   " + "\n"+
                f"μ: ({spatial_metrics['x']['mean']:5.1%}, {spatial_metrics['y']['mean']:5.1%})" + "\n" +
                f"σ: ({spatial_metrics['x']['sd']:5.1%}, {spatial_metrics['y']['sd']:5.1%})" + "\n" +
                f"AREA: {area_pct:^6.2%}",
                fontsize=8, loc="right", fontfamily="monospace",
            )
            # Draw center title
            axs.flat[ind].set_title(
                f"#{hsv_rank[2]+1:.0f}  ",
                fontsize=16, loc="center",fontweight="bold",
            )
        else:  # if cell is empty
            axs.flat[ind].axis("off")  # remove axes for empty cells
            if ind == row_ct*col_ct-1:  # if the last cell
                img_file_num = src_info.get("file_num")
                if img_file_num is not None:
                    dim = "x".join([str(n) for n in img_bgr.shape])
                    """axs.flat[ind].annotate(
                        f"Filename: {img_file_num}.png" + "\n" +
                        f"Race: {img_race_abbr} ({img_race_long})" + "\n" +
                        f"Dim: {dim}" + "\n" +
                        f"K={K}" ,
                        (0, 0), fontsize=16,
                        xycoords="axes fraction",va="bottom",ha="left")"""
                    axs.flat[ind].annotate(
                        f"{img_file_num}.png",
                        (0.5, 0.55), fontsize=24, fontweight="bold",
                        xycoords="axes fraction",va="bottom",ha="center")
                    axs.flat[ind].annotate(
                        f"Dimensions: {dim}" + "\n" +
                        f"Race: {img_race_long}" + "\n" +
                        f"K: {K}" ,
                        (0.5, 0.5), fontsize=12, fontfamily="monospace",
                        xycoords="axes fraction",va="top",ha="center")

        # save axis as singular image
        if ind < 10:
            extent = axs.flat[ind].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.tight_layout()


# # Test running kmeans on different colorspaces and uint8 vs uint32
# Note: float64 is not supported by opencv kmeans so using float32 only

# In[71]:


img_file_num = 14000


# ## Run on HSV when BGR is uint8 -- winner!

# In[15]:


image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
centers, kmeans_masks, res_hsv, clusters = EyeTraumaAnalysis.create_kmeans(img_hsv)
res_bgr = np.uint8(cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR))
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
plot_kmeans_subplots_view_simple(img_bgr, res_bgr, masks_summed)

with pd.option_context("display.precision", 3, "display.width",100):
    print(clusters)


# ## Run on HSV when BGR is float32 from 0-255
# I liked it and originally chose it as a winner to avoid rounding error, but it turned out to be significantly less
# accurate in the per_cluster_mask.png output files.

# In[7]:


image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
# HSV floats are from (0-360,0-1,0-1) while uint8s are 0-255 for all channels
img_hsv = cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
centers, kmeans_masks, res_hsv, clusters = EyeTraumaAnalysis.create_kmeans(img_hsv, max_iter=100)
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
res_bgr = np.uint8(cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR))
#plot_kmeans_subplots_view_simple(img_bgr, res_bgr, masks_summed)
plot_per_cluster_masking(img_bgr, centers, kmeans_masks, src_info={"file_num":img_file_num})

with pd.option_context("display.precision", 3, "display.width",100):
    pass #print(clusters)


# ##  Run on HSV when BGR is float32 from 0-1
# This should work, but the clusters were all weird for some reason.

# In[20]:


np.max(img_hsv, axis=(0,1))


# In[28]:


a=np.float32(img_bgr) /255
img_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
img_hsv = img_hsv / np.array([360,1,1])
np.vstack((np.min(img_hsv, axis=(0,1)), np.max(img_hsv, axis=(0,1))))


# In[ ]:


a=np.float32(img_bgr) / np.array([360,])
np.vstack((np.min(img_hsv, axis=(0,1)), np.max(img_hsv, axis=(0,1))))


# In[19]:


image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
img_hsv = cv2.cvtColor(np.float32(img_bgr)/255.0, cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
centers, kmeans_masks, res_hsv, clusters  = EyeTraumaAnalysis.create_kmeans(img_hsv)
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
res_bgr = np.uint8((cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)) *255)
plot_kmeans_subplots_view_simple(img_bgr, res_bgr, masks_summed)


# In[32]:


np.vstack((np.min(img_hsv, axis=(0,1)), np.max(img_hsv, axis=(0,1))))


# ##  Run on HSV when BGR is float32 from 0-1 and then H is converted from 0-360.0 to 0-1.0

# In[39]:


res_hsv.dtype


# In[3]:


(np.array([360,1,1],dtype="float32")*res_hsv).dtype


# In[36]:


image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
# needs to be float32, not the default float64
img_hsv = cv2.cvtColor(np.float32(img_bgr)/255.0, cv2.COLOR_BGR2HSV) / np.array([360,1,1])
centers, kmeans_masks, res_hsv, clusters  = EyeTraumaAnalysis.create_kmeans(img_hsv)
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
res_bgr = np.uint8((cv2.cvtColor(res_hsv *np.array([360,1,1]), cv2.COLOR_HSV2BGR)) *255)
plot_kmeans_subplots_view_simple(img_bgr, res_bgr, masks_summed)


# ## Run on BGR uint8

# In[18]:


image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
centers, kmeans_masks, res_bgr, clusters = EyeTraumaAnalysis.create_kmeans(img_bgr)
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
plot_kmeans_subplots_view_simple(img_bgr, res_bgr, masks_summed)

clusters[("center","V")] = np.max(centers[["B","G","R"]], axis=1)
clusters = clusters.sort_values(by=("center","V"))
with pd.option_context("display.precision", 3, "display.width",100):
    print(clusters)


# ## Run on old version of function (on HSV when BGR is uint8)

# In[19]:


image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
img_bgr = image.img
# HSV floats are from (0-360,0-1,0-1) while uint8s are 0-255 for all channels
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
centers, ranges, res_hsv, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans_old(img_hsv)
masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
res_bgr = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)
plot_kmeans_subplots_view_simple(img_bgr, res_bgr, masks_summed)

centers["ct"] = np.sum(kmeans_masks, axis=(1,2))
centers["%"] = centers["ct"] / centers["ct"].sum()
print(centers)


# # Run and save on all images

# In[ ]:


print("Saved... ", end="")
for img_file_num in range(14000,14580):
    image = EyeTraumaAnalysis.Image(f"data/01_raw/{img_file_num}.png")
    img_bgr = image.img
    # HSV floats are from (0-360,0-1,0-1) while uint8s are 0-255 for all channels
    img_hsv = cv2.cvtColor( img_bgr, cv2.COLOR_BGR2HSV)  # needs to be float32, not the default float64
    centers, kmeans_masks, res_hsv, clusters = EyeTraumaAnalysis.create_kmeans(img_hsv)
    masks_summed = EyeTraumaAnalysis.get_masked_sums(kmeans_masks)
    res_bgr = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)

    # Save actual kmeans images
    plt.imsave(f"data/02_kmeans/{img_file_num}_kmeans_color.png", np.uint8(res_bgr))
    plt.imsave(f"data/02_kmeans/{img_file_num}_kmeans_hsv.png", np.uint8(res_hsv))
    plt.imsave(f"data/02_kmeans/{img_file_num}_grayscale.png", masks_summed)
    print(img_file_num, end=" ")

    ## Save Clustered View figure
    plot_kmeans_subplots_view(img_bgr, res_bgr, masks_summed)
    plt.savefig(f"outputs/kmeans-clusters/clustered_view/{img_file_num}_clustered_view.png", format="png")
    plt.close()  # close to prevent overconsumption of memory

    ## Save Per Cluster Masking figure
    plot_per_cluster_masking(img_bgr, centers, kmeans_masks)
    plt.savefig(f"outputs/kmeans-clusters/per_cluster_mask/{img_file_num}_per_cluster_mask.png", format="png")
    plt.close()  # close to prevent overconsumption of memory


# # Ethan's old code

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


# In[13]:


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


# In[14]:


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

