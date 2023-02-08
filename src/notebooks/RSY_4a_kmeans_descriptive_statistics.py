#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# # Load data from excel

# In[5]:


kmeans_labels = pd.read_excel("data/01_raw/Ergonautus/Ergonautus_Clusters_Correct_Values.xlsx", dtype={
    "Correct 1":"Int64", # "Int64" is from pandas, unlike int64 and allows null
    "Correct 2":"Int64",
    "Correct 3":"Int64",
    "Borderline":"Int64",
    "Notes":str,
    "Filename":str,
}, na_filter=False) # False na_filters make empty value for str column be "" instead of NaN


# # Calculate metrics

# In[7]:


all_metrics = []
all_kmeans_masks = {}
for ind, filename in enumerate(kmeans_labels["Filename"]):
    img_bgr = skimage.io.imread(os.path.join("data/01_raw/",filename))
    centers, ranges, res_bgr, kmeans_masks = EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
    metrics = EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
    all_metrics.append(metrics)
    all_kmeans_masks[filename] = kmeans_masks

all_metrics = pd.concat(all_metrics, keys=kmeans_labels["Filename"])


# In[15]:


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


# In[16]:


all_metrics_agg = all_metrics.groupby([("Labels","Value")]).agg(["median"])[["Ranks","Values"]]


# # Prepare for plot / Create flat version of df

# In[17]:


all_metrics_flat = all_metrics.copy()
all_metrics_flat.columns = ["-".join(multi_col).rstrip("-") for multi_col in all_metrics.columns]
all_metrics_flat = all_metrics_flat.reset_index()
all_metrics_flat["Ranks-Color-Center-V>4"] = all_metrics_flat["Ranks-Color-Center-V"] >4
all_metrics_flat["Ranks-Color-Center-V>5"] = all_metrics_flat["Ranks-Color-Center-V"] >5


# In[24]:


import plotly.express as px

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
    "Ranks-Color-Mean-V>4": "V Rank >4",
    "Ranks-Color-Mean-V>5": "V Rank >5"
}
color_discrete_map = {
    "True":           px.colors.qualitative.Safe[0], # green
    "Maybe":          px.colors.qualitative.Safe[2], # blue
    "False":          px.colors.qualitative.Safe[1], # red
}
category_orders = {
    "Labels-Value": ["False", "Maybe", "True"],
    }
for var_label_key in var_labels.copy():
    if var_label_key.startswith("Values-"):
        var_label_key_suffix = var_label_key.split("Values-",maxsplit=1)[-1]
        var_labels[f"Ranks-{var_label_key_suffix}"] = var_labels[var_label_key] + " (Rank)"
plotly_template = "plotly_dark"  #"simple_white"


def save_plotly_figure(fig,title):
    fig.write_image("outputs/kmeans-descriptive/" + title + ".png")
    fig.write_html( "outputs/kmeans-descriptive/" + title + ".html",
                        full_html=True,
                        include_plotlyjs="directory" )


# # Plot

# In[25]:


fig = px.box(all_metrics_flat, x="Labels-Value", y="Values-Color-Center-V", points="all",
             boxmode="group", notched=True, width=500, height=300,
             category_orders=category_orders, labels=var_labels, template=plotly_template,
             )
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "Center V box plot"
save_plotly_figure(fig, title)
fig.show()


# In[52]:


fig = px.scatter(all_metrics_flat, x="Values-Color-Center-H", y="Values-Color-Center-V",
                 marginal_x="histogram", marginal_y="histogram",
                 color="Labels-Value", category_orders=category_orders, labels=var_labels, template=plotly_template,)
fig.update_traces(marker=dict(size=2, opacity=0.8),selector=dict(type="scatter"))
fig.show()


# In[26]:


fig = px.scatter_matrix(all_metrics_flat,
                        dimensions=["Values-Color-Center-H", "Values-Color-Center-S", "Values-Color-Center-V",],
                 color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark",)
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "HSV center scatter matrix"
save_plotly_figure(fig, title)
fig.show()


# In[27]:


import plotly.express as px
fig = px.scatter_3d(all_metrics_flat, x="Values-Color-Center-H", y="Values-Color-Center-S",
                    z="Values-Color-Center-V",
                    color="Labels-Value", category_orders=category_orders, labels=var_labels, template=plotly_template,)
fig.update_traces(marker=dict(size=2, opacity=0.8),
                  selector=dict(mode='markers'))
title = "HSV 3D scatter"
save_plotly_figure(fig, title)
fig.show()


# In[92]:


all_metrics_flat_temp = all_metrics_flat.copy()
all_metrics_flat_temp["Ranks-Color-Center-H"] =     all_metrics_flat_temp["Ranks-Color-Center-H"] * 360/10  + np.random.rand(len
                                                                             (all_metrics_flat_temp["Values-Color-Center-H"]))*24-12
all_metrics_flat_temp["Values-Color-Center-H"] = all_metrics_flat_temp["Values-Color-Center-H"] * 360/256
fig = px.scatter_polar(all_metrics_flat_temp, theta="Ranks-Color-Center-H", r="Ranks-Color-Center-S",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark",)
fig.update_traces(marker=dict(size=4, opacity=0.8))
fig.show()


# In[69]:


np.median(all_metrics_flat_temp["Values-Color-Center-H"])


# In[29]:


all_metrics_flat_temp = all_metrics_flat.copy()
all_metrics_flat_temp["Ranks-Color-Center-H"] = all_metrics_flat_temp["Ranks-Color-Center-H"] * 360/10
all_metrics_flat_temp["Values-Color-Center-H"] = all_metrics_flat_temp["Values-Color-Center-H"] * 360/256

fig = px.scatter_polar(all_metrics_flat_temp, theta="Values-Color-Center-H", r="Values-Color-Center-S",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark",)
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "HSV H by S polar scatter"
save_plotly_figure(fig, title)
fig.show()

fig = px.scatter_polar(all_metrics_flat_temp, theta="Values-Color-Center-H", r="Values-Color-Center-V",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark",)
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "HSV H by V polar scatter"
save_plotly_figure(fig, title)
fig.show()


# In[30]:


fig = px.histogram(all_metrics_flat, x="Values-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV H val histogram with box plot"
save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat, x="Values-Color-Center-S", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV S val histogram with box plot"
save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat, x="Values-Color-Center-V", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV V val histogram with box plot"
save_plotly_figure(fig, title)


# In[31]:


fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent", width=500, height=300,
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
title = "HSV H rank histogram with box plot"
save_plotly_figure(fig, title)

fig.show()
fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-S", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent", width=500, height=300,
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
title = "HSV S rank histogram with box plot"
save_plotly_figure(fig, title)

fig.show()
fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-V", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent", width=500, height=300,
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV V rank histogram with box plot"
save_plotly_figure(fig, title)


# In[32]:



fig = px.scatter_matrix(all_metrics_flat,
                        dimensions=["Values-Location-Mean-x", "Values-Location-Mean-y",
                                    "Values-Location-SD-x", "Values-Location-SD-y"],
                        color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark",)
fig.update_traces(marker=dict(size=2, opacity=0.8))
fig.show()
title = "Location scatter matrix"
save_plotly_figure(fig,title)


# In[33]:


fig = px.histogram(all_metrics_flat, x="Values-Location-Mean-x", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- Mean x"
save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat, x="Values-Location-Mean-y", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- Mean y"
save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat, x="Values-Location-SD-x", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- SD x"
save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat, x="Values-Location-SD-y", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- SD y"
save_plotly_figure(fig, title)


# In[124]:


all_metrics.agg(["median","max"])

