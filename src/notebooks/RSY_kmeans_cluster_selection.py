#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import importlib
import json
import numpy as np
import pandas as pd
import scipy.ndimage as snd
import skimage
import uuid

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


# In[3]:


import labelbox  # labelbox.Client, MALPredictionImport, LabelImport
import labelbox.schema.ontology # OntologyBuilder, Tool, Classification, Option
import labelbox.data.annotation_types
""" from labelbox.data.annotation_types import (
    Label, ImageData, ObjectAnnotation, MaskData,
    Rectangle, Point, Line, Mask, Polygon,
    Radio, Checklist, Text,
    ClassificationAnnotation, ClassificationAnswer
)"""
import labelbox.data.serialization # NDJsonConverter
import labelbox.schema.media_type # MediaType

import labelbox.schema.queue_mode # QueueMode


# In[10]:


# Labelbox API stored in separate file since it is specific for a labelbox
#account and shouldn't be committed to git. Contact the
# team (i.e. Rahul Yerrabelli) in order to access to the data on your own account.
with open("auth/LABELBOX_API_KEY.json", "r") as infile:
  json_data = json.load(infile)
LB_API_KEY = json_data["API_KEY"]
del json_data   # delete sensitive info

PROJECT_ID = "clds7rw8a17bd07140ida09o9"
DATASET_ID = "cldscwrp0071k07zf84smghrw"
ONTOLOGY_ID = "cldsdg4re1xo707xnbvnadmuw"

client = labelbox.Client(api_key=LB_API_KEY)
del LB_API_KEY   # delete sensitive info
project = client.get_project(PROJECT_ID)
dataset_lb = client.get_dataset(DATASET_ID)


# In[ ]:


mask_annotation = labelbox.data.annotation_types.ObjectAnnotation(
  name = "mask", # must match your ontology feature's name
  value = labelbox.data.annotation_types.Mask(mask=mask_data, color=color),
)


# In[342]:


def create_kmeans(img_bgr, K=10):  # K is number of clusters
    #np.all(skimage.io.imread("data/01_raw/14579.png") == skimage.io.imread(data_row.row_data))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Z_hsv = img_hsv.reshape((-1,3))
    # convert to np.float32
    Z_hsv = np.float32(Z_hsv)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,centers=cv2.kmeans(Z_hsv,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    res_hsv = centers[label.flatten()]
    res_hsv2 = res_hsv.reshape(img_hsv.shape)
    res_bgr = cv2.cvtColor(res_hsv2, cv2.COLOR_HSV2BGR)
    # res2 = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)


    # sort centers by HSV "value" - aka sort by grayscale
    centers = centers[centers[:, 2].argsort()]

    #centers_indices = np.argsort(centers, axis=0)   # sorts each column separately

    kmeans_masks = []
    for ind in range(K):
        # Can use opencv in range or kmeans
        #kmeans_masks.append(cv2.inRange(res_hsv2, centers[ind], centers[ind]))
        kmeans_masks.append( np.all(res_hsv2 == centers[ind], axis=-1) )
        #kmeans_masks.append( res_hsv2==centers[ind])
    kmeans_masks = np.array(kmeans_masks)

    # can't make centers a DataFrame until now since needed numpy for opencv in range or numpy comparison
    centers = pd.DataFrame(centers, columns=["H","S","V"])
    mins = np.array([np.min(img_hsv[kmeans_mask],axis=0) for kmeans_mask in kmeans_masks])
    maxs = np.max([np.min(img_hsv[kmeans_mask],axis=0) for kmeans_mask in kmeans_masks])
    ranges = pd.DataFrame(maxs - mins, columns=["H","S","V"])
    return centers, ranges, res_bgr, kmeans_masks

def get_spatial_metrics(mask):
    # scipy can perform the mean (center of mass), but not the standard deviation
    # spatial_means = snd.center_of_mass(mask)
    x = np.linspace(0, 1, mask.shape[1])
    y = np.linspace(0, 1, mask.shape[0])
    xgrid, ygrid = np.meshgrid(x, y)
    grids = {"x": xgrid, "y":ygrid}
    to_return = {"x":{}, "y":{}}
    for ind, grid in grids.items():
        to_return[ind]["mean"] = np.mean(grids[ind], where=mask.astype(bool))
        to_return[ind]["sd"] = np.std(grids[ind], where=mask.astype(bool))
    return to_return


def get_kmeans_metrics(centers, ranges, kmeans_masks):
    spatial_metrics_list = [get_spatial_metrics(kmeans_mask) for kmeans_mask in kmeans_masks]
    spatial_metrics_pd = pd.concat([pd.DataFrame({
        "x": [spatial_metrics["x"]["mean"] for spatial_metrics in spatial_metrics_list],
        "y": [spatial_metrics["y"]["mean"] for spatial_metrics in spatial_metrics_list],}),
        pd.DataFrame({
        "x": [spatial_metrics["x"]["sd"] for spatial_metrics in spatial_metrics_list],
        "y": [spatial_metrics["y"]["sd"] for spatial_metrics in spatial_metrics_list],
    })], axis=1, keys=["Mean","SD"])

    area_fractions = pd.DataFrame([np.count_nonzero(kmeans_mask)/np.prod(kmeans_mask.shape) for kmeans_mask in
                               kmeans_masks], columns=pd.MultiIndex.from_tuples([("","")]))
    color_metrics = pd.concat([centers, ranges], axis=1, keys=["Center","Range"])

    all_metrics = pd.concat([color_metrics, spatial_metrics_pd, area_fractions], axis=1,
                            keys=["Color","Location","Area"])
    all_metrics_ranks = np.argsort(all_metrics, axis=0) + 1

    return pd.concat([all_metrics, all_metrics_ranks], axis=1, keys=["Values","Ranks"])

def choose_kmeans_cluster(metrics):
    metrics = metrics.copy()
    metrics[("Values","Location","SD","x y")] = metrics[
        [("Values","Location","SD","x"),
         ("Values","Location","SD","y")]].max(axis=1) # get max of x and y SD
    likely = metrics[
        (metrics["Ranks"]["Color"]["Center"]["V"] >= 5) &
        (metrics["Values"]["Location"]["Mean"]["x"] >= 0.3) &
        (metrics["Values"]["Location"]["Mean"]["x"] <= 0.7) &
        (metrics["Values"]["Location"]["Mean"]["y"] >= 0.3) &
        (metrics["Values"]["Location"]["Mean"]["y"] <= 0.7) &
        (metrics["Values"]["Location"]["SD"]["x"] <= 0.25) &
        (metrics["Values"]["Location"]["SD"]["y"] <= 0.25)
    ]
    # trim down further
    if likely.shape[0] > 2:
        likely = likely.sort_values(by=("Values","Location","SD","x y"))[:2]
    return likely


# In[343]:


centers, ranges, res_bgr, kmeans_masks = create_kmeans(img_bgr)
metrics = get_kmeans_metrics(centers, ranges, kmeans_masks)
chosen = choose_kmeans_cluster(metrics)
# get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
# to be included in the combined mask
kmeans_masks_chosen = np.any(kmeans_masks[chosen.index],axis=0)


# In[392]:


label = []
kmeans_masks_chosens = []

for ind, data_row in enumerate(dataset_lb.export_data_rows()):

    # read the image from the cloud
    img_bgr = skimage.io.imread(data_row.row_data)
    # Confirmed that the image from Labelbox is identical to the local image
    #np.all(skimage.io.imread(f"data/01_raw/{data_row.external_id}.png") == img_bgr)

    # apply kmeans on the image and choose the best cluster
    centers, ranges, res_bgr, kmeans_masks = create_kmeans(img_bgr)
    metrics = get_kmeans_metrics(centers, ranges, kmeans_masks)
    chosen = choose_kmeans_cluster(metrics)
    # get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
    # to be included in the combined mask
    kmeans_masks_chosen = np.any(kmeans_masks[chosen.index],axis=0)
    kmeans_masks_chosens.append(kmeans_masks_chosen)

    # Append to list of suggested labels
    # astype converts from bool to uin8. Has to be uint8 not np.int64; otherwise throws an error since 0-255 not
    # guaranteed
    lb_mask_data = labelbox.data.annotation_types.MaskData.from_2D_arr(kmeans_masks_chosen.astype("uint8")*255)
    color = (28, 230, 255)
    lb_mask_annotation = labelbox.data.annotation_types.ObjectAnnotation(
      name = "Conjunctiva", # must match your ontology feature's name
      value = labelbox.data.annotation_types.Mask(mask=lb_mask_data, color=color),
    )
    label.append(labelbox.data.annotation_types.Label(
        data=labelbox.data.annotation_types.ImageData(uid=data_row.uid),
        annotations = [ lb_mask_annotation ]
    ))

    if ind > 3:
        break


# Convert our label from a Labelbox class object to the underlying NDJSON format required for upload
label_ndjson = list(labelbox.data.serialization.NDJsonConverter.serialize(label))


# In[412]:


ontologies = client.get_ontologies(name_contains="eye")


# In[419]:


for ont in ontologies:
    print(ont)

