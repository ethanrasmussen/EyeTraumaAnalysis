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


# In[2]:


importlib.reload(EyeTraumaAnalysis);


# In[8]:


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


# In[9]:


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


# In[11]:


centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
metrics = EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
chosen = EyeTraumaAnalysis.kmeans.choose_kmeans_cluster(metrics)
# get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
# to be included in the combined mask
kmeans_masks_chosen = np.any(kmeans_masks[chosen.index],axis=0)


# In[12]:


labels = []
kmeans_masks_chosens = []

for ind, data_row in enumerate(dataset_lb.export_data_rows()):

    # read the image from the cloud
    img_bgr = skimage.io.imread(data_row.row_data)
    # Confirmed that the image from Labelbox is identical to the local image
    #np.all(skimage.io.imread(f"data/01_raw/{data_row.external_id}.png") == img_bgr)

    # apply kmeans on the image and choose the best cluster
    centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
    metrics = EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
    chosen = EyeTraumaAnalysis.kmeans.choose_kmeans_cluster(metrics)
    # get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
    # to be included in the combined mask
    kmeans_masks_chosen = np.any(kmeans_masks[chosen.index],axis=0)
    kmeans_masks_chosens.append(kmeans_masks_chosen)

    # Append to list of suggested labels
    # astype converts from bool to uin8. Has to be uint8 not np.int64; otherwise throws an error since 0-255 not
    # guaranteed
    lb_mask_data = labelbox.data.annotation_types.MaskData.from_2D_arr(kmeans_masks_chosen.astype("uint8")*255)

    color = (28, 230, 255)   # 1CE6FF

    lb_mask_data = labelbox.data.annotation_types.MaskData(arr= np.zeros(img_bgr.shape[:2] +(3,),dtype='uint8'))


    lb_mask_annotation = labelbox.data.annotation_types.ObjectAnnotation(
      name = "Conjunctiva", # must match your ontology feature's name
      value = labelbox.data.annotation_types.Mask(mask=lb_mask_data, color=color),
    )

    labels.append(labelbox.data.annotation_types.Label(
        data=labelbox.data.annotation_types.ImageData(uid=data_row.uid),
        annotations = [ lb_mask_annotation ]
    ))

    if ind > 1:
        break


# In[13]:


labels[0].annotations[0].value.mask.arr.shape


# In[14]:


labels[0].data


# In[19]:


radio_annotation = labelbox.data.annotation_types.ClassificationAnnotation(
  name="health",
  value=labelbox.data.annotation_types.Radio(answer = labelbox.data.annotation_types.ClassificationAnswer(name =
                                                                                                          "healthy"))
)


# In[16]:



# Convert our label from a Labelbox class object to the underlying NDJSON format required for upload
labels_ndjson = list(labelbox.data.serialization.NDJsonConverter.serialize(labels))


# In[30]:


labels_ndjson


# In[18]:


# Upload MAL label for this data row in project
upload_job = labelbox.MALPredictionImport.create_from_objects(
    client = client,
    project_id = project.uid,
    name="mal_job"+str(uuid.uuid4()),
    predictions=labels_ndjson)

print("Errors:", upload_job.errors)


# In[7]:


labels[0].annotations[0].value.mask


# In[425]:


ontologies = client.get_ontologies(name_contains="eye")


# In[419]:


for ont in ontologies:
    print(ont)

