#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import sys
import importlib
import json
import numpy as np
import pandas as pd
import scipy
import scipy.ndimage as snd
import skimage
import uuid
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2
import plotly
import plotly.express as px
import plotly.graph_objects as go


if os.getcwd().split("/")[-1] == "notebooks":  # if cwd is located where this file is
    os.chdir("../..")  # go two folders upward (the if statement prevents error if cell is rerun)
directory_path = os.path.abspath(os.path.join("src"))
if directory_path not in sys.path:
    sys.path.append(directory_path)
print(directory_path)
import src.EyeTraumaAnalysis


# In[6]:


importlib.reload(src.EyeTraumaAnalysis);


# # Load metrics

# In[7]:


get_ipython().system('pip3 install pickle5')
import pickle5 as pickle


# In[8]:


# reload data as pickle protocol 4
with open("../../data/03_first_25percent_metrics/color_and_spatial_metrics" + ".pkl", "rb") as fh:
  data = pickle.load(fh)
  data.to_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics" + "_p4" + ".pkl")
with open("../../data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + ".pkl", "rb") as fh:
  data = pickle.load(fh)
  data.to_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + "_p4" + ".pkl")
with open("../../data/03_first_25percent_metrics/color_and_spatial_metrics_agg" + ".pkl", "rb") as fh:
  data = pickle.load(fh)
  data.to_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_agg" + "_p4" + ".pkl")


# In[9]:


# all_metrics = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics" + ".pkl")
# all_metrics_flat = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + ".pkl")
# all_metrics_agg = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_agg" + ".pkl")
all_metrics = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics" + "_p4" + ".pkl")
all_metrics_flat = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + "_p4" + ".pkl")
all_metrics_agg = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_agg" + "_p4" + ".pkl")


# 

# In[11]:


def save_plotly_figure(fig: plotly.graph_objs.Figure, title: str, directory="outputs/kmeans-descriptive-subsets/"):
    fig.write_image(os.path.join(directory, title + ".png"))
    fig.write_html( os.path.join(directory, title + ".html"),
                    full_html=True, include_plotlyjs="directory" )


# In[12]:


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

# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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

