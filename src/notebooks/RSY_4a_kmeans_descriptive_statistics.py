#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[5]:


import os
import sys
import importlib
import json
import numpy as np
import pandas as pd
import scipy.ndimage as snd
import skimage
import uuid

if os.getcwd().split("/")[-1] == "notebooks":  # if cwd is located where this file is
    os.chdir("../..")  # go two folders upward (the if statement prevents error if cell is rerun)
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

import plotly.express as px
import plotly


# # Load metrics

# In[7]:


all_metrics = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_p4" + ".pkl")
all_metrics_flat = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_flat_p4" + ".pkl")
all_metrics_agg = pd.read_pickle("../../data/03_first_25percent_metrics/color_and_spatial_metrics_agg_p4" + ".pkl")


# # Alternatively, recalculate metrics

# ## Load data from excel

# In[9]:


kmeans_labels = pd.read_excel("../../data/01_raw/Ergonautus/Ergonautus_Clusters_Correct_Values.xlsx", dtype={
    "Correct 1":"Int64", # "Int64" is from pandas, unlike int64 and allows null
    "Correct 2":"Int64",
    "Correct 3":"Int64",
    "Borderline":"Int64",
    "Notes":str,
    "Filename":str,
}, na_filter=False) # False na_filters make empty value for str column be "" instead of NaN


# ## Calculate metrics

# In[12]:


all_metrics = []
all_kmeans_masks = {}
for ind, filename in enumerate(kmeans_labels["Filename"]):
    img_bgr = skimage.io.imread(os.path.join("../../data/01_raw/",filename))
    centers, ranges, res_bgr, kmeans_masks = src.EyeTraumaAnalysis.kmeans.create_kmeans(img_bgr)
    metrics = src.EyeTraumaAnalysis.kmeans.get_kmeans_metrics(centers, ranges, kmeans_masks)
    all_metrics.append(metrics)
    all_kmeans_masks[filename] = kmeans_masks

all_metrics = pd.concat(all_metrics, keys=kmeans_labels["Filename"])


# In[13]:


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


# ## Create aggregate version of metrics df

# In[14]:


all_metrics_agg = all_metrics.groupby([("Labels","Value")]).agg(["median"])[["Ranks","Values"]]

all_metrics_agg = all_metrics.groupby([("Labels","Value")]).agg(["min","median","max"])
all_metrics_agg.to_excel("../../outputs/kmeans-descriptive/aggregate.xlsx")


# ## Create flat version of metrics df

# In[15]:


all_metrics_flat = all_metrics.copy()
all_metrics_flat.columns = ["-".join(multi_col).rstrip("-") for multi_col in all_metrics.columns]
all_metrics_flat = all_metrics_flat.reset_index()

# Create tuples of values i.e. "...-HSV" instead of just "...-H"
current_columns = all_metrics_flat.columns.copy()
for col in current_columns:
    suffix = "-H"
    if col.endswith(suffix):
        col_prefix = col[0:-len(suffix)]  # get column name up to suffix
        # create a column that has the values as a tuple
        all_metrics_flat[col_prefix + "-HSV"] = list(zip(
            all_metrics_flat[col_prefix+"-H"],
            all_metrics_flat[col_prefix+"-S"],
            all_metrics_flat[col_prefix+"-V"],
        ))

    suffix = "-x"
    if col.endswith(suffix):
        col_prefix = col[0:-len(suffix)]  # get column name up to suffix

        # check if column is integers
        if pd.api.types.is_integer_dtype(all_metrics_flat[col_prefix + suffix]):
            # create a column that has the values as a tuple
            all_metrics_flat[col_prefix + "-xy"] = list(zip(
                all_metrics_flat[col_prefix+"-x"],
                all_metrics_flat[col_prefix+"-y"],
            ))
        else:
            # create a column that  formatted strings - easier to view if numbers are originally floats instead of
            # integers
            all_metrics_flat[col_prefix + "-xy"] =                 all_metrics_flat[col_prefix+"-x"].map("{:.1%}, ".format) +                 all_metrics_flat[col_prefix+"-y"].map("{:.1%}".format)

for ind in range(1,10+1):
    all_metrics_flat[f"Ranks-Color-Center-H>={ind}"] = all_metrics_flat["Ranks-Color-Center-H"] >=ind
for ind in range(1,10+1):
    all_metrics_flat[f"Ranks-Color-Center-S>={ind}"] = all_metrics_flat["Ranks-Color-Center-S"] >=ind
for ind in range(1,10+1):
    all_metrics_flat[f"Ranks-Color-Center-V>={ind}"] = all_metrics_flat["Ranks-Color-Center-V"] >=ind

for val_cutoff in [100]:
    all_metrics_flat[f"Values-Color-Center-H>={val_cutoff}"] = all_metrics_flat["Values-Color-Center-H"] >=val_cutoff
for val_cutoff in range(135,175+1,5):
    all_metrics_flat[f"Values-Color-Center-S>={val_cutoff}"] = all_metrics_flat["Values-Color-Center-S"] >=val_cutoff
for val_cutoff in range( 75,150+1,5):
    all_metrics_flat[f"Values-Color-Center-V>={val_cutoff}"] = all_metrics_flat["Values-Color-Center-V"] >=val_cutoff


# Get Hue in terms of 360 degrees
K=10
angle_per_K = 360/10
all_metrics_flat["Ranks-Color-Center-H360"] = all_metrics_flat["Ranks-Color-Center-H"] * angle_per_K                                                + np.random.rand(len(all_metrics_flat["Values-Color-Center-H"]))                                              *angle_per_K*0.75-angle_per_K*0.75/2
all_metrics_flat["Values-Color-Center-H360"] = all_metrics_flat["Values-Color-Center-H"] * 360/256


# ## Save values so they don't have to be rerun every time

# In[ ]:


all_metrics.to_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics" + ".pkl")
all_metrics_flat.to_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + ".pkl")
all_metrics_agg.to_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics_agg.pkl")

all_metrics.to_excel("data/03_first_25percent_metrics/color_and_spatial_metrics" + ".xlsx")
all_metrics_flat.to_excel("data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + ".xlsx")
all_metrics_agg.to_excel("data/03_first_25percent_metrics/color_and_spatial_metrics_agg" + ".xlsx")


# # Prepare for plotting

# In[ ]:





# In[16]:


def save_plotly_figure(fig: plotly.graph_objs.Figure, title: str, directory="outputs/kmeans-descriptive/"):
    fig.write_image(os.path.join(directory, title + ".png"))
    fig.write_html( os.path.join(directory, title + ".html"),
                    full_html=True, include_plotlyjs="directory" )


# In[17]:


color_discrete_map = {
    "True":           px.colors.qualitative.Plotly[2], # green
    "Maybe":          px.colors.qualitative.Plotly[0], # blue
    "False":          px.colors.qualitative.Plotly[1], # red
}
pattern_shape_map = {}
category_orders = {
    "Labels-Value": ["False", "Maybe", "True"],
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
    "Ranks-Color-Mean-V>4": "V Rank >4",
    "Ranks-Color-Mean-V>5": "V Rank >5"
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

plotly_template = "plotly_dark"  #"simple_white"


# # Plot

# In[ ]:


fig = px.box(all_metrics_flat, x="Labels-Value", y="Values-Color-Center-V", points="all",
             boxmode="group", notched=True, width=500, height=300,
             category_orders=category_orders, labels=var_labels, template=plotly_template
             )
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "Center V box plot"
save_plotly_figure(fig, title)
fig.show()


# In[ ]:


fig = px.scatter(all_metrics_flat, x="Values-Color-Center-H", y="Values-Color-Center-V",
                 marginal_x="histogram", marginal_y="histogram",
                 hover_data=point_hover_data,
                 color="Labels-Value", color_discrete_map=color_discrete_map,
                 category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_traces(marker=dict(size=2, opacity=0.8), selector=dict(type="scattergl"))
fig.for_each_trace( lambda trace: trace.update(marker=dict(size=2, opacity=0.8)) if trace.type == "scattergl" else (), )
fig.show()


# In[ ]:


fig = px.scatter_matrix(all_metrics_flat,
                        dimensions=["Values-Color-Center-H", "Values-Color-Center-S", "Values-Color-Center-V",],
                        hover_data=point_hover_data,
                        color="Labels-Value", color_discrete_map=color_discrete_map,
                        category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "HSV scatter matrix- HSV center"
save_plotly_figure(fig, title)
fig.show()


# In[ ]:


fig = px.scatter_3d(all_metrics_flat,
                    x="Values-Color-Center-H", y="Values-Color-Center-S", z="Values-Color-Center-V",
                    hover_data=point_hover_data,
                    color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.for_each_trace( lambda trace: trace.update(marker=dict(size=2, opacity=0.8),) if trace.mode == "markers" else (), )
title = "HSV 3D scatter- HSV center"
save_plotly_figure(fig, title)
fig.show()


# In[ ]:


fig = px.scatter_polar(all_metrics_flat, theta="Ranks-Color-Center-H360", r="Ranks-Color-Center-S",
                    hover_data=point_hover_data,
                    color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)

fig.update_traces(marker=dict(size=4, opacity=0.8))
fig.show()


# In[ ]:


fig = px.scatter_polar(all_metrics_flat, theta="Values-Color-Center-H", r="Values-Color-Center-S",
                       hover_data=point_hover_data,
                    color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "HSV polar scatter- H by S"
save_plotly_figure(fig, title)
fig.show()

fig = px.scatter_polar(all_metrics_flat, theta="Values-Color-Center-H", r="Values-Color-Center-V",
                   hover_data=point_hover_data,
                    color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_traces(marker=dict(size=2, opacity=0.8))
title = "HSV polar scatter- H by S"
save_plotly_figure(fig, title)
fig.show()


# In[ ]:


fig = px.histogram(all_metrics_flat, x="Values-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                    color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- H val"
save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat, x="Values-Color-Center-S", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                    color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- S val"
save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat, x="Values-Color-Center-V", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                    color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- V val"
save_plotly_figure(fig, title)


# In[ ]:


fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-H", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent", width=500, height=300,
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- H rank"
save_plotly_figure(fig, title)

fig.show()
fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-S", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent", width=500, height=300,
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- S rank"
save_plotly_figure(fig, title)

fig.show()
fig = px.histogram(all_metrics_flat, x="Ranks-Color-Center-V", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent", width=500, height=300,
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                    category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "HSV histogram with box plot- V rank"
save_plotly_figure(fig, title)


# In[ ]:


fig = px.scatter_matrix(all_metrics_flat,
                        dimensions=["Values-Location-Mean-x", "Values-Location-Mean-y",
                                    "Values-Location-SD-x", "Values-Location-SD-y"],
                        color="Labels-Value", color_discrete_map=color_discrete_map,
                        category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_traces(marker=dict(size=2, opacity=0.8))
fig.show()
title = "Location scatter matrix"
save_plotly_figure(fig,title)


# In[ ]:





# In[19]:


all_metrics_flat2 = all_metrics_flat[all_metrics_flat["Ranks-Color-Center-V"] >= 4]

fig = px.histogram(all_metrics_flat2, x="Values-Location-Mean-x", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                   category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- Mean x"
# save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat2, x="Values-Location-Mean-y", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                   category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- Mean y"
# save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat2, x="Values-Location-SD-x", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", color_discrete_map=color_discrete_map,
                   category_orders=category_orders, labels=var_labels, template=plotly_template)
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- SD x"
# save_plotly_figure(fig, title)

fig = px.histogram(all_metrics_flat2, x="Values-Location-SD-y", marginal="box", opacity=0.6,
                   barmode="group",  histnorm="percent",
                   color="Labels-Value", category_orders=category_orders, labels=var_labels, template="plotly_dark")
fig.update_layout(bargap=0.04)
fig.update_layout(font=dict(family="Arial",size=16,), margin=dict(l=20, r=20, t=20, b=20))
fig.show()
title = "Location histogram with box plot- SD y"
# save_plotly_figure(fig, title)


# In[ ]:




