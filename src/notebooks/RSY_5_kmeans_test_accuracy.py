#!/usr/bin/env python
# coding: utf-8

# In[84]:


import os
import sys
import importlib
import json
import numpy as np
import pandas as pd
import scipy.ndimage as snd
import skimage
import uuid

if os.getcwd().split("/")[-1] == "notebooks":
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


# In[85]:


kmeans_labels = pd.read_excel("data/01_raw/Ergonautus/Ergonautus_Clusters_Correct_Values.xlsx", dtype={
    "Correct 1":"Int64", # "Int64" is from pandas, unlike int64 and allows null
    "Correct 2":"Int64",
    "Correct 3":"Int64",
    "Borderline":"Int64",
    "Notes":str,
    "Filename":str,
}, na_filter=False) # False na_filters make empty value for str column be "" instead of NaN


# In[144]:


all_metrics = []
all_kmeans_masks = {}
for ind, filename in enumerate(kmeans_labels["Filename"]):
    img_bgr = skimage.io.imread(os.path.join("data/01_raw/",filename))
    centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
    metrics = EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
    all_metrics.append(metrics)
    all_kmeans_masks[filename] = kmeans_masks

all_metrics = pd.concat(all_metrics, keys=kmeans_labels["Filename"])


# In[292]:


all_metrics.loc[:, ("Labels","Value","","")] = "False"
all_metrics.loc[:, ("Labels","Correct","","")] = False
all_metrics.loc[:, ("Labels","Borderline","","")] = False
for ind, row in kmeans_labels.iterrows():
    filename = row["Filename"]

    correct_indices =row[["Correct 1","Correct 2","Correct 3"]] - 1   # subtract by 1 to go from 1-10 to 0-9
    correct_indices = correct_indices[~pd.isnull(correct_indices)].to_numpy().astype(int)
    file_maskrow_correct_indices = [(filename,correct_index) for correct_index in correct_indices]
    all_metrics.loc[file_maskrow_correct_indices, ("Labels","Correct","","")]=True
    all_metrics.loc[file_maskrow_correct_indices, ("Labels","Value","","")]="True"

    borderline_indices = row[["Borderline"]] - 1
    borderline_indices = borderline_indices[~pd.isnull(borderline_indices)].to_numpy().astype(int)
    file_maskrow_borderline_indices = [(filename,borderline_index) for borderline_index in borderline_indices]
    all_metrics.loc[file_maskrow_borderline_indices, ("Labels","Borderline","","")]=True
    all_metrics.loc[file_maskrow_borderline_indices, ("Labels","Value","","")]="Maybe"

# Reorder
all_metrics = all_metrics[["Labels","Ranks","Values"]]
# Label the second index
all_metrics.index.names = [all_metrics.index.names[0], "Mask"]


# In[151]:


all_metrics_agg = all_metrics.groupby([("Labels","Value")]).agg(["median"])[["Ranks","Values"]]


# # Plot

# In[293]:


import plotly.express as px
#fig = px.box(all_metrics, x=("Labels","Value","",""), y=("Values","Color","Center","V"))
fig = px.box(y=all_metrics[("Values","Color","Center","V")].values,
             x=all_metrics[("Labels","Value")].values, points="all", notched=True, boxmode="group")
fig.show()


# # Create flat version of df

# In[301]:


var_labels = {
    "Labels-Value": "Conjunctiva cluster",
    "Values-Color-Center-H": "Middle H",
    "Values-Color-Center-S": "Middle S",
    "Values-Color-Center-V": "Middle V",
}
plotly_template = "plotly_dark"


# In[294]:


all_metrics_flat = all_metrics.copy()
all_metrics_flat.columns = ["-".join(multi_col).rstrip("-") for multi_col in all_metrics.columns]
all_metrics_flat = all_metrics_flat.reset_index()
all_metrics_flat.columns


# In[302]:


import plotly.express as px
#fig = px.box(all_metrics, x=("Labels","Value","",""), y=("Values","Color","Center","V"))
fig = px.box(all_metrics_flat, ("Labels-Value"), "Values-Color-Center-V", points="all",
             boxmode="group", notched=True, width=500, height=300, template=plotly_template, labels=var_labels)
fig.show()


# In[303]:


import plotly.express as px
fig = px.scatter_matrix(all_metrics_flat,
                        dimensions=["Values-Color-Center-H", "Values-Color-Center-S", "Values-Color-Center-V",],
                        color="Labels-Value", template=plotly_template, labels=var_labels)
fig.update_traces(marker=dict(size=2))
fig.show()


# In[306]:


import plotly.express as px
fig = px.scatter_3d(all_metrics_flat, x="Values-Color-Center-H", y="Values-Color-Center-S",
                    z="Values-Color-Center-V", color="Labels-Value", template=plotly_template, labels=var_labels)
fig.update_traces(marker=dict(size=2, opacity=0.5),
                  selector=dict(mode='markers'))
fig.show()


# In[307]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter_matrix(all_metrics_flat,
                        dimensions=["Values-Location-Mean-x", "Values-Location-Mean-y",
                                    "Values-Location-SD-x", "Values-Location-SD-y"],
                        color="Labels-Value", template=plotly_template, labels=var_labels)
fig.update_traces(marker=dict(size=2))
fig.show()


# In[268]:


import plotly.express as px
df = px.data.wind()
all_metrics_flat_temp = all_metrics_flat.copy()
all_metrics_flat_temp["Ranks-Color-Center-H"] = all_metrics_flat_temp["Ranks-Color-Center-H"] * 360/10
all_metrics_flat_temp["Values-Color-Center-H"] = all_metrics_flat_temp["Values-Color-Center-H"] * 360/256
fig = px.bar_polar(all_metrics_flat_temp, theta="Ranks-Color-Center-H", r="Ranks-Color-Center-S", color="Labels-Value", template="plotly_dark")
fig.show()


# In[124]:


all_metrics.agg(["median","max"])


# In[88]:


all_metrics


# # Test model

# In[76]:


segmentations_trues = {}
clusters_trues = {}

for ind, row in kmeans_labels.iterrows():
    correct_indices =row[["Correct 1","Correct 2","Correct 3"]] - 1   # subtract by 1 to go from 1-10 to 0-9
    correct_indices = correct_indices[~pd.isnull(correct_indices)].to_numpy().astype(int)
    filename = row["Filename"]
    kmeans_masks = all_kmeans_masks[filename]
    # get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
    # to be included in the combined mask
    segmentations_trues[filename] =  np.any(kmeans_masks[correct_indices],axis=0)
    clusters_trues[filename] =  correct_indices


# In[147]:


segmentations_preds = {}
clusters_preds = {}

for ind, filename in enumerate(kmeans_labels["Filename"]):

    metrics = all_metrics.loc[filename]
    kmeans_masks = all_kmeans_masks[filename]
    chosen = EyeTraumaAnalysis.kmeans.choose_kmeans_cluster(metrics)

    # get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
    # to be included in the combined mask
    segmentations_preds[filename] =  np.any(kmeans_masks[chosen.index],axis=0)
    clusters_preds[filename] =  np.array(chosen.index)


# In[15]:


all_metrics


# In[145]:


def calculate_prediction_statistics_areas(segmentations_trues, segmentations_preds):
    prediction_statistics = {}
    for filename in segmentations_trues.keys():
        mask_true = segmentations_trues[filename]
        mask_pred = segmentations_preds[filename]

        total_area = np.prod(mask_true.shape)

        intersection_area = true_positive_area = np.count_nonzero(mask_true & mask_pred) / total_area
        false_positive_area = np.count_nonzero(mask_true & ~mask_pred) / total_area
        false_negative_area = np.count_nonzero(~mask_true & mask_pred) / total_area
        true_negative_area = np.count_nonzero(~mask_true & ~mask_pred) / total_area
        union_area = true_positive_area + false_positive_area + false_negative_area

        # https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
        # Jaccard's index: Intersection over union
        jaccard_area = intersection_area / union_area
        # dice index: Jaccard's index but double counting intersection
        dice_area = 2*intersection_area / (union_area + intersection_area)

        prediction_statistics[filename] = {
            "jaccard":jaccard_area,
            "dice": dice_area,
            "true_positive": true_positive_area,
            "false_positive": false_positive_area,
            "false_negative": false_negative_area,
            "true_negative": true_negative_area,
            "union": union_area,
        }


# In[149]:


def calculate_prediction_statistics_clusters(clusters_true, clusters_pred, total_clusters=10):
    prediction_statistics = {}
    for filename in segmentations_trues.keys():
        clusters_true = clusters_trues[filename]
        clusters_pred = clusters_preds[filename]

        #intersection_clusters = true_positive_clusters = len([cluster for cluster in clusters_true if cluster in clusters_pred])
        intersection_clusters             = true_positive_clusters = len([cluster for cluster in range(total_clusters)
                                            if cluster in clusters_true and cluster in clusters_pred])
        false_positive_clusters = len([cluster for cluster in range(total_clusters)
                                       if cluster not in clusters_true and cluster in clusters_pred])
        false_negative_clusters = len([cluster for cluster in range(total_clusters)
                                       if cluster in clusters_true and cluster not in clusters_pred])
        true_negative_clusters  = len([cluster for cluster in range(total_clusters)
                                       if cluster not in clusters_true and cluster not in clusters_pred])
        union_clusters = true_positive_clusters + false_positive_clusters + false_negative_clusters

        # https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
        # Jaccard's index: Intersection over union
        jaccard_clusters = intersection_clusters , union_clusters
        # dice index: Jaccard's index but double counting intersection
        dice_clusters = 2*intersection_clusters , (union_clusters + intersection_clusters)
        distribution_clusters = (true_positive_clusters, false_positive_clusters, false_negative_clusters, true_negative_clusters)
        prediction_statistics[filename] = jaccard_clusters

        prediction_statistics[filename] = {
            "jaccard":jaccard_clusters,
            "dice": dice_clusters,
            "true_positive": true_positive_clusters,
            "false_positive": false_positive_clusters,
            "false_negative": false_negative_clusters,
            "true_negative": true_negative_clusters,
            "union": union_clusters,
        }

    return prediction_statistics


# In[151]:


prediction_statistic_area = calculate_prediction_statistics_areas(segmentations_trues, segmentations_preds)
prediction_statistic_clusters = calculate_prediction_statistics_clusters(clusters_trues, clusters_preds)
[prediction_statistic["jaccard"] for prediction_statistic in prediction_statistic_clusters.values()]


# In[11]:


all_metrics = pd.concat(all_metrics, keys=kmeans_labels["Filename"])


# In[ ]:


centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
metrics = EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
chosen = EyeTraumaAnalysis.kmeans.choose_kmeans_cluster(metrics)
# get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
# to be included in the combined mask
kmeans_masks_chosen = np.any(kmeans_masks[chosen.index],axis=0)


# In[2]:


os.getcwd()


# In[ ]:




