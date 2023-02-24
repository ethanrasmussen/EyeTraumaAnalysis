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


# reload package if there are any changes
importlib.reload(EyeTraumaAnalysis)


# In[2]:


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


# In[3]:


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


# In[4]:


kmeans_labels = pd.read_excel("data/01_raw/Ergonautus/Ergonautus_Clusters_Correct_Values.xlsx", dtype={
    "Correct 1":"Int64", # "Int64" is from pandas, unlike int64 and allows null
    "Correct 2":"Int64",
    "Correct 3":"Int64",
    "Borderline":"Int64",
    "Notes":str,
    "Filename":str,
}, na_filter=False) # False na_filters make empty value for str column be "" instead of NaN


# In[5]:


all_metrics = []
all_kmeans_masks = {}
for ind, filename in enumerate(kmeans_labels["Filename"]):
    img_bgr = skimage.io.imread(os.path.join("data/01_raw/",filename))
    centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
    metrics = EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
    all_metrics.append(metrics)
    all_kmeans_masks[filename] = kmeans_masks

all_metrics = pd.concat(all_metrics, keys=kmeans_labels["Filename"])


# In[8]:


filename


# In[7]:


plt.imshow(img_bgr)


# In[175]:


centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
len(kmeans_masks)


# In[92]:


# Test model
segmentations_trues = {}
clusters_trues = {}

for ind, row in kmeans_labels.iterrows():
    correct_indices = row[["Correct 1", "Correct 2", "Correct 3"]] - 1  # subtract by 1 to go from 1-10 to 0-9
    correct_indices = correct_indices[~pd.isnull(correct_indices)].to_numpy().astype(int)
    filename = row["Filename"]
    kmeans_masks = all_kmeans_masks[filename]
    # get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
    # to be included in the combined mask
    segmentations_trues[filename] = np.any(kmeans_masks[correct_indices], axis=0)
    clusters_trues[filename] = correct_indices


# In[6]:


(kmeans_masks[(1,2)]).shape


# In[166]:


np.max(np.concatenate(list(clusters_trues.values())))


# In[172]:


clusters_trues["14436.png"]


# In[3]:


plt.imshow(segmentations_trues["14436.png"])


# In[185]:



filename = "14436.png"
# using the variable axs for multiple Axes
fig, axs = plt.subplots(3, 4)
for ind, ax in enumerate(axs.flat):
    if ind <len(all_kmeans_masks[filename]):
        ax.imshow(all_kmeans_masks[filename][ind])
    else:
        ax.axis("off")


# In[94]:


for ind, data_row in enumerate(dataset_lb.export_data_rows()):
    print(data_row.external_id)
    break


# In[106]:


# Create urls to mask data for upload
def signing_function(obj_bytes: bytes) -> str:
    url = client.upload_data(content=obj_bytes, sign=True)
    return url


#labels = []
labels1 = labelbox.data.annotation_types.LabelList()  # doesn't really matter if LabelList or regular list
kmeans_masks_chosens = []

for ind, data_row in enumerate(dataset_lb.export_data_rows()):
    filename = data_row.external_id
    image_num_str = filename.split(".")[0].split("_")[0]
    image_num = int(image_num_str)

    if filename not in segmentations_trues:
        continue
    else:
        print(filename)

    # read the image from the cloud
    img_bgr = skimage.io.imread(data_row.row_data)
    # Confirmed that the image from Labelbox is identical to the local image
    #print( np.all(skimage.io.imread(f"data/01_raw/{filename}") == img_bgr) )

    # apply kmeans on the image and choose the best cluster
    centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
    metrics = EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
    chosen = EyeTraumaAnalysis.kmeans.choose_kmeans_cluster(metrics)
    # get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
    # to be included in the combined mask
    #kmeans_masks_chosen = np.any(kmeans_masks[chosen.index],axis=0)
    kmeans_masks_chosen = segmentations_trues[filename]
    kmeans_masks_chosens.append(kmeans_masks_chosen)

    # Append to list of suggested labels
    # astype converts from bool to uin8. Has to be uint8 not np.int64; otherwise throws an error since 0-255 not
    # guaranteed
    lb_mask_data = labelbox.data.annotation_types.MaskData.from_2D_arr(kmeans_masks_chosen.astype("uint8")*255)

    conj_color_label = (28, 230, 255)   # 1CE6FF

    #mask_data = np.zeros(img_bgr.shape[:2] +(3,),dtype='uint8')
    mask_data = np.where(kmeans_masks_chosens[-1][...,np.newaxis],conj_color_label,0).astype("uint8")
    lb_mask_data = labelbox.data.annotation_types.MaskData(arr= mask_data)


    lb_mask_annotation = labelbox.data.annotation_types.ObjectAnnotation(
      name = "conjunctiva", # must match your ontology feature's name
      value = labelbox.data.annotation_types.Mask(mask=lb_mask_data, color=conj_color_label),
    )

    labels1.append(labelbox.data.annotation_types.Label(
        data=labelbox.data.annotation_types.ImageData(uid=data_row.uid),
        annotations = [ lb_mask_annotation ]
    ))
    labels1[-1].add_url_to_masks(signing_function)

    if len(labels1) >= 8:
        break


labels_ndjson1 = list(labelbox.data.serialization.NDJsonConverter.serialize(labels1))


# In[104]:



# Upload MAL label for this data row in project
upload_job = labelbox.MALPredictionImport.create_from_objects(
    client = client,
    project_id = project.uid,
    name="mal_job"+str(uuid.uuid4()),
    predictions=labels_ndjson1
)

print("Errors:", upload_job.errors)


# In[144]:



# Upload label for this data row in project
upload_job = labelbox.LabelImport.create_from_objects(
    client = client,
    project_id = project.uid,
    name="label_import_job"+str(uuid.uuid4()),
    labels=labels_ndjson1)

print("Errors:", upload_job.errors)
print(" ")


# In[145]:


dataset_lb.delete_labels()


# In[49]:


#labels2 = []
labels2 = labelbox.data.annotation_types.LabelList()

for ind, data_row in enumerate(dataset_lb.export_data_rows()):

    filename = data_row.external_id
    image_num_str = filename.split(".")[0].split("_")[0]
    image_num = int(image_num_str)

    if   14000 <= image_num <= 14103:  # 104 images
        race = "asian"
    elif 14104 <= image_num <= 14289:  # 186 images
        race = "black"
    elif 14290 <= image_num <= 14393:  # 104 images
        race="hispanic"
    elif 14394 <= image_num <= 14579:  # 186 images
        race="white (non-hispanic)"
    else:
        race = "unknown"

    if   14000 <= image_num <= 14999:
        # all images are either left or were flipped LR to become by the original authors
        side = "left"

    if image_num < 1000:
        health = "diseased"
    else:
        health = "healthy"

    annotations = []

    # Python annotation
    annotations.append(labelbox.data.annotation_types.ClassificationAnnotation(
        name="race",
        value=labelbox.data.annotation_types.Radio(
            answer=labelbox.data.annotation_types.ClassificationAnswer(
                name=race
            )
        )
    ))
    # NDJSON
    #radio_annotation_ndjson = {"name": "race","answer": {"name": race}}

    annotations.append(labelbox.data.annotation_types.ClassificationAnnotation(
        name="health",
        value=labelbox.data.annotation_types.Radio(
            answer=labelbox.data.annotation_types.ClassificationAnswer(
                name=health
            )
        )
    ))
    annotations.append(labelbox.data.annotation_types.ClassificationAnnotation(
        name="side",
        value=labelbox.data.annotation_types.Radio(
            answer=labelbox.data.annotation_types.ClassificationAnswer(
                name=side
            )
        )
    ))


    labels2.append(labelbox.data.annotation_types.Label(
        data=labelbox.data.annotation_types.ImageData(uid=data_row.uid),
        annotations = annotations
    ))



# In[28]:


labels2[0].annotations


# In[50]:


labels_ndjson2 = list(labelbox.data.serialization.NDJsonConverter.serialize(labels2))

# Upload label for this data row in project
upload_job = labelbox.LabelImport.create_from_objects(
    client = client,
    project_id = project.uid,
    name="label_import_job"+str(uuid.uuid4()),
    labels=labels_ndjson2)

print("Errors:", upload_job.errors)
print(" ")


# In[34]:


for ind, data_row in enumerate(dataset_lb.export_data_rows()):
    print(data_row)
    break


# In[35]:


labels = project.export_labels(download=True)


# In[36]:


labels


# In[38]:



# Convert our label from a Labelbox class object to the underlying NDJSON format required for upload
labels_ndjson = list(labelbox.data.serialization.NDJsonConverter.serialize(labels))


# In[39]:


ontologies = client.get_ontologies(name_contains="eye")


# In[40]:


for ont in ontologies:
    print(ont)


# In[146]:


downloaded_labels = []
ct = 0
for downloaded_label in project.label_generator():
    ct = ct + 1
    downloaded_labels.append(downloaded_label.annotations)
    print(downloaded_label.annotations)
print(ct)


# In[147]:


[downloaded_label[0].name for downloaded_label in downloaded_labels]


# In[148]:


for downloaded_label in downloaded_labels:
    print([downloaded_label_ann.name for downloaded_label_ann in downloaded_label])


# In[142]:


counts = {}
for downloaded_label in downloaded_labels:
    a = tuple(downloaded_label_ann.name for downloaded_label_ann in downloaded_label)
    if a in counts.keys():
        counts[a] = counts[a] + 1
    else:
        counts[a] = 1
print(counts)


# In[149]:


ct


# In[133]:


downloaded_label


# In[120]:


downloaded_label.annotations[0].value


# In[151]:


labels2[0]


# In[125]:


downloaded_label.annotations[-1]


# In[124]:


downloaded_label.annotations[0].extra["value"]


# In[126]:


ct


# In[ ]:




