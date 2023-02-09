#!/usr/bin/env python
# coding: utf-8

# In[143]:


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

import plotly.express as px
import plotly


# # Load metrics

# In[144]:


all_metrics = pd.read_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics" + ".pkl")
all_metrics_flat = pd.read_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + ".pkl")
all_metrics_agg = pd.read_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics_agg" + ".pkl")


# # Create function for calculating roc metrics

# In[183]:





# In[221]:


def calculate_roc(truths, predict_scores, true_value=None):
    if true_value is None:
        truths = np.array(truths, dtype=bool)
    elif isinstance(true_value,(list,dict,tuple,set,pd.Series,pd.DataFrame,np.ndarray)):
        truths = np.array(truths) #== true_value  # check if values equal the whole collection
        truths_temp = np.zeros(truths.shape, dtype=bool)  # start with false and then apply the | (or) operator
        for each_true_value in true_value:  # check if values equal any of the elements in the collection
            truths_temp = truths_temp | (truths == each_true_value)
        truths = truths_temp
    else:
        truths = np.array(truths) == true_value
    predict_scores = np.array(predict_scores)
    predict_scores_sorted = np.sort(np.unique(predict_scores))
    #thresholds = np.concatenate( ([np.min(predict_scores)-0.01], predict_scores) )
    thresholds = np.concatenate( (
        [predict_scores_sorted[0]-0.01],
        (predict_scores_sorted[1:]+predict_scores_sorted[:-1])/2,
        [predict_scores_sorted[-1]+0.01]
    ))
    predictions = predict_scores >= thresholds[...,np.newaxis]
    # predictions has one more dimension than predict_scores
    true_pos  =  truths &  predictions
    false_pos = ~truths &  predictions
    false_neg =  truths & ~predictions
    true_neg  = ~truths & ~predictions

    true_pos_ct  = np.count_nonzero(true_pos,  axis=-1)
    false_pos_ct = np.count_nonzero(false_pos, axis=-1)
    false_neg_ct = np.count_nonzero(false_neg, axis=-1)
    true_neg_ct  = np.count_nonzero(true_neg,  axis=-1)

    # Below is a good paper to review the formulas
    # https://www.frontiersin.org/articles/10.3389/fpubh.2017.00307/full
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

    return pd.DataFrame({
        "threshold": thresholds,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "1-specificity": fpr,
    })


# In[197]:


np.equal( np.array([0,0,1,1,1]), [1])


# # Prepare for plotting

# In[145]:


def save_plotly_figure(fig: plotly.graph_objs.Figure, title: str):
    fig.write_image("outputs/kmeans-descriptive-subsets/" + title + ".png")
    fig.write_html( "outputs/kmeans-descriptive-subsets/" + title + ".html",
                        full_html=True, include_plotlyjs="directory" )


# In[166]:


color_discrete_map = {
    "True":           px.colors.qualitative.Plotly[2], # green
    "Maybe":          px.colors.qualitative.Plotly[0], # blue
    "False":          px.colors.qualitative.Plotly[1], # red
}
pattern_shape_map = {}
category_orders = {
    "Labels-Value": ["False", "Maybe", "True"],
    "facet_col": [False, True],
    "facet_row": [False, True],
}

# This is only the start. It will be added to programmatically later
var_labels = {
    "Labels-Value": "Conjunctiva cluster",
    "Values-Color-Center-H": "Center H",
    "Values-Color-Center-S": "Center S",
    "Values-Color-Center-V": "Center V",
    "Values-Color-Range-H": "Range H",
    "Values-Color-Range-S": "Range S",
    "Values-Color-Range-V": "Range V",
    "Values-Location-Mean-x": "Mean x",
    "Values-Location-Mean-y": "Mean y",
    "Values-Location-SD-x": "SD x",
    "Values-Location-SD-y": "SD y",
}

var_labels_copy = var_labels.copy()
suffixes = ["-H","-x"]
for var_label_key in var_labels_copy:
    for suffix in suffixes:
        if var_label_key.endswith(suffix):
            sep = suffix[:1]  # should be "-"
            suffix_letter = suffix[1:]  # should be "-H" or "-x"
            # Get name up to suffix letter e.g. "Values-Color-Center-"
            var_label_key_prefix = var_label_key[0:-len(suffix_letter)]
            # Get all possible suffixes for the prefix i.e. "H", "S", "V"
            suffix_letter_options = [var_label_key[len(var_label_key_prefix):] for var_label_key in var_labels_copy
                                          if var_label_key.startswith(var_label_key_prefix)]
            combined_suffix_letters = "".join(suffix_letter_options)
            # Get combined value
            var_label_val_prefix = var_labels[var_label_key_prefix + suffix_letter][:-len(suffix_letter)]
            combined_var_label_key = var_label_key_prefix + combined_suffix_letters
            combined_var_label_val = var_label_val_prefix + combined_suffix_letters
            var_labels[combined_var_label_key] = combined_var_label_val


# Add labels for ranks
var_labels_copy = var_labels.copy()
for var_label_key in var_labels_copy:
    if var_label_key.startswith("Values-"):
        var_label_key_suffix = var_label_key.split("Values-",maxsplit=1)[-1]
        var_labels[f"Ranks-{var_label_key_suffix}"] = var_labels[var_label_key] + " (Rank)"

# Add labels
for var_label_key in all_metrics_flat.columns:
    for comparator in [">","<"]:
        if comparator in var_label_key:
            stem, comparison = var_label_key.split(comparator, maxsplit=1)
            if stem in var_labels:
                var_labels[var_label_key] =                     (var_labels[stem] + comparator + comparison).replace(">=","≥").replace("<=","≤")
            else:
                print(var_label_key, stem)
                print(var_labels_copy)
                raise KeyError

#point_hover_data = ["Values-Color-Center-HSV","Ranks-Color-Center-HSV",
#                    "Values-Location-Mean-xy","Ranks-Location-Mean-xy",
#                    "Values-Location-SD-xy","Ranks-Location-SD-xy"]
point_hover_data = {
    "Values-Color-Center-H": False,
    "Values-Color-Center-S": False,
    "Values-Color-Center-V": False,
    "Ranks-Color-Center-H": False,
    "Ranks-Color-Center-S": False,
    "Ranks-Color-Center-V": False,
    "Values-Color-Center-HSV":True,
    "Ranks-Color-Center-HSV":True,
    "Values-Location-Mean-xy":True,
    "Ranks-Location-Mean-xy":True,
    "Values-Location-SD-xy":True,
    "Ranks-Location-SD-xy":True,
}
roc_hover_data = {
    "sensitivity":":0.2%",
    "1-specificity":":0.2%",
    "threshold":True
}

plotly_template = "plotly_dark"  #"simple_white"


# # Plot

# In[32]:


fig = px.histogram(all_metrics_flat, x="Values-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group", histnorm="percent",
                   facet_col="Values-Color-Center-H>=100",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.update_xaxes(matches=None)
fig.for_each_xaxis(lambda axis: axis.update(showticklabels=True))
fig.show()
title = "HSV histogram with box plot- H val split at >=100"
save_plotly_figure(fig, title)


# In[33]:


fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",
                   facet_col="Values-Color-Center-V>=75",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- H rank split at V val>75"
save_plotly_figure(fig, title)


# In[35]:


fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",
                   facet_col="Ranks-Color-Center-V>=4",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- H val split at V rank>=4"
save_plotly_figure(fig, title)


# In[37]:


fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",
                   facet_col="Ranks-Color-Center-V>=5",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- H val split at V rank>=5"
save_plotly_figure(fig, title)


# In[40]:


fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",
                   facet_col="Ranks-Color-Center-V>=6",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                   category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- H val split at V rank>=6"
save_plotly_figure(fig, title)


# In[44]:


fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",
                   facet_col="Values-Color-Center-V>=75",
                   facet_row="Values-Color-Center-S>=155",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                   category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram- H val split at V >=75 and S >=75"
save_plotly_figure(fig, title)


# In[208]:


roc_df = calculate_roc(all_metrics_flat["Labels-Value"],
                       all_metrics_flat["Values-Color-Center-V"], true_value="True")
fig = px.area(roc_df,
              x="1-specificity", y="sensitivity",
              hover_data=roc_hover_data,
                 category_orders=category_orders, labels=var_labels, template=plotly_template,
)
fig.add_shape( type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1 )
fig.show()
title = "HSV ROC curve - V value"
save_plotly_figure(fig, title)


# In[222]:


roc_df = calculate_roc(all_metrics_flat["Labels-Value"],
                       all_metrics_flat["Ranks-Color-Center-V"], true_value="True")
fig = px.area(roc_df,
              x="1-specificity", y="sensitivity",
              hover_data=roc_hover_data, text="threshold",
                 category_orders=category_orders, labels=var_labels, template=plotly_template,
)
fig.add_shape( type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1 )
fig.show()
title = "HSV ROC curve - V rank"
save_plotly_figure(fig, title)


# In[224]:


roc_df = calculate_roc(all_metrics_flat["Labels-Value"],
                       all_metrics_flat["Ranks-Color-Center-H"], true_value="True")
roc_df.sort_values(by="1-specificity")
fig = px.line(roc_df,
              x="1-specificity", y="sensitivity",
              hover_data=roc_hover_data, text="threshold", markers=True,
              range_y=[0,1], range_x=[0,1],
                 category_orders=category_orders, labels=var_labels, template=plotly_template,
)
fig.add_shape( type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1 )
fig.show()
title = "HSV ROC curve - H rank"
save_plotly_figure(fig, title)

roc_df = calculate_roc(all_metrics_flat["Labels-Value"],
                       all_metrics_flat["Ranks-Color-Center-S"], true_value="True")
roc_df.sort_values(by="1-specificity")
fig = px.line(roc_df,
              x="1-specificity", y="sensitivity",
              hover_data=roc_hover_data, text="threshold", markers=True,
              range_y=[0,1], range_x=[0,1],
                 category_orders=category_orders, labels=var_labels, template=plotly_template,
)
fig.add_shape( type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1 )
fig.show()
title = "HSV ROC curve - S rank"
save_plotly_figure(fig, title)

roc_df = calculate_roc(all_metrics_flat["Labels-Value"],
                       all_metrics_flat["Ranks-Color-Center-V"], true_value="True")
roc_df.sort_values(by="1-specificity")
fig = px.line(roc_df,
              x="1-specificity", y="sensitivity",
              hover_data=roc_hover_data, text="threshold", markers=True,
              range_y=[0,1], range_x=[0,1],
                 category_orders=category_orders, labels=var_labels, template=plotly_template,
)
fig.add_shape( type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1 )
fig.show()
title = "HSV ROC curve - V rank"
save_plotly_figure(fig, title)


# In[59]:


np.concatenate([[1],[2]])


# In[123]:


a = np.array([0,5,10])
b = np.array([6,5,4,3])
c = a >= b[...,np.newaxis]
c


# In[104]:


(~np.array([True,False,True,False]) & c)


# In[141]:


np.array(pd.Series([1,2,3,"False"]),dtype=bool)


# In[136]:





# In[190]:


calculate_roc([0,0,1,1,1],[0.1,0.2,0.3,0.4,0.5],true_value=["a"])


# In[ ]:




