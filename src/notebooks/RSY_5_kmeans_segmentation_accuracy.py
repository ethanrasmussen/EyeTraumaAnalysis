#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

import src.EyeTraumaAnalysis

print(directory_path)
importlib.reload(src.EyeTraumaAnalysis);
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2


# In[3]:


kmeans_labels = pd.read_excel("C:/Users/ethan/PycharmProjects/EyeTraumaAnalysis/data/01_raw/Ergonautus/Ergonautus_Clusters_Correct_Values.xlsx", dtype={
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
    img_bgr = skimage.io.imread(os.path.join("C:/Users/ethan/PycharmProjects/EyeTraumaAnalysis/data/01_raw/",filename))
    centers, ranges, res_bgr, kmeans_masks = src.EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
    metrics = src.EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
    all_metrics.append(metrics)
    all_kmeans_masks[filename] = kmeans_masks

all_metrics = pd.concat(all_metrics, keys=kmeans_labels["Filename"])


# In[6]:


all_metrics_agg = all_metrics.groupby([("Labels","Value")]).agg(["median"])[["Ranks","Values"]]


# In[7]:


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


# In[8]:


segmentations_preds = {}
clusters_preds = {}
for ind, filename in enumerate(kmeans_labels["Filename"]):
    metrics = all_metrics.loc[filename]
    kmeans_masks = all_kmeans_masks[filename]
    chosen = src.EyeTraumaAnalysis.kmeans.choose_kmeans_cluster(metrics)

    # get combined masks of the clusters chosen. The .any() applies an OR so only a pixel needs to be in only one cluster
    # to be included in the combined mask
    segmentations_preds[filename] = np.any(kmeans_masks[chosen.index], axis=0)
    clusters_preds[filename] = np.array(chosen.index)


# In[9]:


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
        dice_area = 2 * intersection_area / (union_area + intersection_area)

        prediction_statistics[filename] = {
            "jaccard"       : jaccard_area,
            "dice"          : dice_area,
            "true_positive" : true_positive_area,
            "false_positive": false_positive_area,
            "false_negative": false_negative_area,
            "true_negative" : true_negative_area,
            "union"         : union_area,
        }

    prediction_statistics = pd.DataFrame.from_dict(prediction_statistics, orient="index")
    return prediction_statistics


def calculate_prediction_statistics_clusters(clusters_true, clusters_pred, total_clusters=10):
    prediction_statistics = {}
    for filename in segmentations_trues.keys():
        clusters_true = clusters_trues[filename]
        clusters_pred = clusters_preds[filename]

        #intersection_clusters = true_positive_clusters = len([cluster for cluster in clusters_true if cluster in clusters_pred])
        intersection_clusters = true_pos_clusters = [cluster for cluster in range(total_clusters)
                                                              if cluster in clusters_true and cluster in clusters_pred]
        false_pos_clusters = [cluster for cluster in range(total_clusters)
                                       if cluster not in clusters_true and cluster in clusters_pred]
        false_neg_clusters = [cluster for cluster in range(total_clusters)
                                       if cluster in clusters_true and cluster not in clusters_pred]
        true_neg_clusters = [cluster for cluster in range(total_clusters)
                                      if cluster not in clusters_true and cluster not in clusters_pred]
        union_clusters = true_pos_clusters + false_pos_clusters + false_neg_clusters

        true_pos_ct = len(true_pos_clusters)
        true_neg_ct = len(true_neg_clusters)
        false_neg_ct = len(false_neg_clusters)
        false_pos_ct = len(false_pos_clusters)

        accuracy = true_pos_ct + true_neg_ct / ( true_pos_ct + false_pos_ct + false_neg_ct + true_neg_ct )
        # Sensitivity aka Recall aka True positive rate (TPR)
        sensitivity = tpr = true_pos_ct / ( true_pos_ct + false_neg_ct )
        # Specificity aka True negative rate (TNR)
        specificity = tnr = true_neg_ct / ( true_neg_ct + false_pos_ct )
        # Positive predictive value (PPV) aka Precision
        ppv = true_pos_ct / ( true_pos_ct + false_pos_ct )
        # Negative predictive value (NPV)
        npv = true_neg_ct / ( true_neg_ct + false_neg_ct )
        # False discovery rate (FDR)
        fdr = 1 - ppv
        # False omission rate (FOR, called FOMR in code)
        fomr = 1 - npv
        # False negative rate (FNR)
        fnr = 1 - tpr
        # False positive rate (FPR) aka 1-specificity
        fpr = 1 - tnr

        # https://towardsdatascience.com/how-accurate-is-image-segmentation-dd448f896388
        # Jaccard's index: Intersection over union
        jaccard_clusters = intersection_clusters, union_clusters
        # dice index: Jaccard's index but double counting intersection
        dice_clusters = 2 * intersection_clusters, (union_clusters + intersection_clusters)
        distribution_clusters = (
        true_positive_clusters, false_positive_clusters, false_negative_clusters, true_negative_clusters)
        prediction_statistics[filename] = jaccard_clusters

        prediction_statistics[filename] = {
            "jaccard_raw"       : jaccard_clusters,
            "jaccard"  : jaccard_clusters[0]/jaccard_clusters[1],
            "dice_raw"          : dice_clusters,
            "TP" : true_positive_clusters,
            "FP": false_positive_clusters,
            "FN": false_negative_clusters,
            "TN" : true_negative_clusters,
            "U"         : union_clusters,
        }

    prediction_statistics = pd.DataFrame.from_dict(prediction_statistics, orient="index")
    return prediction_statistics


# In[10]:


prediction_statistic_area = calculate_prediction_statistics_areas(segmentations_trues, segmentations_preds)
prediction_statistic_clusters = calculate_prediction_statistics_clusters(clusters_trues, clusters_preds)

#clusters_jaccards = [prediction_statistic["jaccard"] for prediction_statistic in prediction_statistic_clusters
# .values()]


# In[11]:


prediction_statistic_clusters["jaccard_raw"].value_counts(sort=True, normalize=False)


# In[12]:


prediction_statistic_clusters["jaccard"].agg(["mean","median","std","min","max"])


# In[ ]:




