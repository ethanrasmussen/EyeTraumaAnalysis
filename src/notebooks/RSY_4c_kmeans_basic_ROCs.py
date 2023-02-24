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

import EyeTraumaAnalysis
from EyeTraumaAnalysis import calculate_roc


# In[8]:


os.getcwd()


# In[9]:


importlib.reload(EyeTraumaAnalysis);


# # Load metrics

# In[11]:


# Load metrics
all_metrics = pd.read_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics" + ".pkl")
all_metrics_flat = pd.read_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics_flat" + ".pkl")
all_metrics_agg = pd.read_pickle("data/03_first_25percent_metrics/color_and_spatial_metrics_agg" + ".pkl")


# # Test function for calculating roc metrics

# In[41]:


importlib.reload(EyeTraumaAnalysis.kmeans)
roc_df, auc, comparator = calculate_roc(all_metrics_flat["Labels-Value"],
                                        all_metrics_flat["Ranks-Color-Center-S"], true_value="True")
auc, comparator


# # Prepare for plotting

# In[5]:


def save_plotly_figure(fig: plotly.graph_objs.Figure, title: str, directory="outputs/kmeans-basic-rocs/"):
    fig.write_image(os.path.join(directory, title + ".png"))
    fig.write_html( os.path.join(directory, title + ".html"),
                    full_html=True, include_plotlyjs="directory" )


# In[14]:


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

# This is only the start of var_labels. It will be added to programmatically later
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
    "specificity":":0.2%",
    #"1-specificity":":0.2%",
    "threshold":True
}

plotly_template = "plotly_dark"  #"simple_white"


# In[48]:


def customize_roc_curve(fig: plotly.graph_objs.Figure, add_reference_line=True):
    if add_reference_line:
        fig.add_shape(type="line", line=dict(dash="dash", width=2), x0=1, y0=0, x1=0, y1=1)
    fig.update_layout(
        template="simple_white", title=title,
        font=dict(
                family="Arial",
                size=16,
                color="black",
            ),
        xaxis=dict(
            zeroline=True,
            range=[1,0], # reversed range. Alternatively, fig.update_xaxes(autorange="reversed")
            showgrid=True,
            title="Specificity (reversed)",
            nticks=20,
            mirror="ticks",
            gridcolor="#DDD",
            showspikes=True, spikemode="across", spikethickness=2, spikedash="solid"
        ),
        yaxis=dict(
            zeroline=True,
            range=[0,1],
            showgrid=True,
            title="Sensitivity",
            nticks=20,
            mirror="ticks",
            gridcolor="#DDD",
            showspikes=True, spikemode="across", spikethickness=2, spikedash="solid"
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bordercolor="Black", #font_size=16,
            borderwidth=2,
        ),
        autosize=False,

    )


# In[310]:


def add_threshold_annotations(fig: plotly.graph_objs.Figure, roc_df, comparator: str, color=None):
    if color is None:
        color = fig.data[0].line.color
    for ind, row in roc_df.iterrows():
        if ind==0 or ind==roc_df.shape[0] - 1:  # if first or last row, then skip
            continue
        fig.add_annotation(
            x=roc_df.loc[ind, "specificity"],
            y=roc_df.loc[ind, "sensitivity"],
            text=f"{comparator}{roc_df.loc[ind, 'threshold']}",
            arrowhead=2,
            font=dict(color=color),
            #arrowcolor=fig.data[0].line.color,
            bgcolor="#eee", bordercolor="#000", opacity=0.8,
        )

def add_auc_annotation(fig: plotly.graph_objs.Figure, auc):
    fig.add_annotation(
        xanchor="right",yanchor="bottom",
        x=0.01, y=0.01, borderpad=5,
        text=f"<b>AUC: {auc:.3f}</b>",
        font=dict(size=16),
        showarrow=False,
        opacity=0.8,
        bgcolor="#FFF", bordercolor="#000",
        borderwidth=2,
    )


# # Plot

# In[311]:


title = "HSV ROC curve - V value"
predictor_name = "Values-Color-Center-V"
roc_df, auc, comparator = calculate_roc(all_metrics_flat["Labels-Value"],
                               all_metrics_flat[predictor_name], true_value="True")
fig = px.area(roc_df,
              x="specificity", y="sensitivity",
              hover_data=roc_hover_data, markers=True, title=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}",
              category_orders=category_orders, labels=var_labels, template=plotly_template,
              )
customize_roc_curve(fig)
add_auc_annotation(fig, auc)
fig.show()
save_plotly_figure(fig, title)


# In[312]:


title = "HSV ROC curve - V rank"
predictor_name = "Ranks-Color-Center-V"
roc_df, auc, comparator = calculate_roc(all_metrics_flat["Labels-Value"],
                                        all_metrics_flat[predictor_name], true_value="True")
fig = px.area(roc_df,
              x="specificity", y="sensitivity",
              hover_data=roc_hover_data, markers=True, title=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}",
              category_orders=category_orders, labels=var_labels, template=plotly_template,
              )
add_threshold_annotations(fig, roc_df, comparator)
add_auc_annotation(fig, auc)

customize_roc_curve(fig)
fig.update_layout(font=dict(family="Arial",size=16,))  #, margin=dict(l=20, r=20, t=20, b=20)
fig.show()
save_plotly_figure(fig, title)


# In[313]:


for title, predictor_name in zip(
        ["HSV ROC curve - H rank", "HSV ROC curve - S rank", "HSV ROC curve - V rank"],
        ["Ranks-Color-Center-H", "Ranks-Color-Center-S", "Ranks-Color-Center-V"]):
    roc_df, auc, _ = calculate_roc(all_metrics_flat["Labels-Value"],
                                   all_metrics_flat[predictor_name], true_value="True")
    fig = px.area(roc_df,
                  x="specificity", y="sensitivity",
                  hover_data=roc_hover_data, markers=True,
                  range_y=[0, 1], range_x=[0, 1], title=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}",
                  category_orders=category_orders, labels=var_labels, template=plotly_template,
                  )
    add_threshold_annotations(fig, roc_df, comparator)
    add_auc_annotation(fig, auc)
    customize_roc_curve(fig)
    fig.show()
    save_plotly_figure(fig, title)


# In[314]:


for title, predictor_name in zip(
        ["HSV ROC curve - H", "HSV ROC curve - S", "HSV ROC curve - V"],
        ["Values-Color-Center-H", "Values-Color-Center-S", "Values-Color-Center-V"]):
    roc_df, auc, _ = calculate_roc(all_metrics_flat["Labels-Value"],
                                   all_metrics_flat[predictor_name], true_value="True")
    fig = px.area(roc_df,
                  x="specificity", y="sensitivity",
                  hover_data=roc_hover_data, markers=True,
                  range_y=[0, 1], range_x=[0, 1], title=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}",
                  category_orders=category_orders, labels=var_labels, template=plotly_template,
                  )
    customize_roc_curve(fig)
    add_auc_annotation(fig, auc)
    fig.show()
    save_plotly_figure(fig, title)


# In[315]:



title = "HSV ROC Curve - HSV centers"
predictor_names = ["Values-Color-Center-H", "Values-Color-Center-S", "Values-Color-Center-V"]

fig = go.Figure()
# Add diagonal random chance reference line
fig.add_shape(type="line", line=dict(dash="dot",width=2), x0=1, y0=0, x1=0, y1=1)

for ind, predictor_name in enumerate(predictor_names):
    roc_df, auc, comparator = calculate_roc(all_metrics_flat["Labels-Value"],
                                   all_metrics_flat[predictor_name], true_value="True")
    comparator = comparator.replace(">=","≥").replace("<=","≤")
    youden = roc_df["specificity"] + roc_df["sensitivity"] -1
    max_youden_loc = youden.argmax()
    color = px.colors.qualitative.Safe[ind]
    fig.add_trace(go.Scatter(
        x=roc_df["specificity"], y=roc_df["sensitivity"],
        mode="lines+markers", opacity=0.75,
        name=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}"
    ))
    print(roc_df["specificity"][max_youden_loc], roc_df["sensitivity"][max_youden_loc],)
    fig.add_annotation(
        x=roc_df["specificity"][max_youden_loc], y=roc_df["sensitivity"][max_youden_loc],
        text=f"At {comparator}{roc_df['threshold'][max_youden_loc]}",
        showarrow=True,
        font=dict(color=color),
        opacity=0.8,
        bgcolor="#FFF", bordercolor="#000",
        arrowwidth=2,
        arrowhead=1)
customize_roc_curve(fig)
fig.show()
save_plotly_figure(fig, title)


# In[319]:


title = "HSV ROC Curve - HSV centers (rank)"
predictor_names = ["Ranks-Color-Center-H", "Ranks-Color-Center-S", "Ranks-Color-Center-V"]

fig = go.Figure()
# Add diagonal random chance reference line
fig.add_shape(type="line", line=dict(dash="dot",width=2), x0=1, y0=0, x1=0, y1=1)

textpositions = ["bottom right", "top right", "top left",]
for ind, predictor_name in enumerate(predictor_names):
    roc_df, auc, comparator = calculate_roc(all_metrics_flat["Labels-Value"],
                                   all_metrics_flat[predictor_name], true_value="True")
    comparator = comparator.replace(">=","≥").replace("<=","≤")
    youden = roc_df["specificity"] + roc_df["sensitivity"] -1
    max_youden_loc = youden.argmax()
    color = px.colors.qualitative.Safe[ind]
    fig.add_trace(go.Scatter(
        x=roc_df["specificity"], y=roc_df["sensitivity"],
        mode="lines+markers+text", opacity=0.75,
        #text=roc_df["threshold"].apply( (comparator+"{:}").format), textposition=textpositions[ind],
        name=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}",
        marker=dict(color=color),
    ))
    fig.add_annotation(
        x=roc_df["specificity"][max_youden_loc], y=roc_df["sensitivity"][max_youden_loc],
        text=f"At {comparator}{roc_df['threshold'][max_youden_loc]}",
        showarrow=True,
        font=dict(color=color),
        #arrowcolor=color,
        bgcolor="#FFF", bordercolor="#000",
        opacity=0.8,
        arrowwidth=2,
        arrowhead=2)
    max_youden_loc = 1
    fig.add_annotation(
        x=roc_df["specificity"][max_youden_loc], y=roc_df["sensitivity"][max_youden_loc],
        text=f"At {comparator}{roc_df['threshold'][max_youden_loc]}",
        showarrow=True,
        font=dict(color=color),
        #arrowcolor=color,
        bgcolor="#FFF", bordercolor="#000",
        opacity=0.8,
        arrowwidth=2,
        arrowhead=2)

customize_roc_curve(fig)
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
save_plotly_figure(fig, title)


# In[ ]:





# In[322]:


import plotly.graph_objects as go

title = "Loc ROC Curve - xy value"
predictor_names = ["Values-Location-Mean-x", "Values-Location-Mean-y",
                   "Values-Location-SD-x", "Values-Location-SD-y"]

fig = go.Figure()
# Add diagonal random chance reference line
fig.add_shape(type="line", line=dict(dash="dot",width=2), x0=1, y0=0, x1=0, y1=1)

for ind, predictor_name in enumerate(predictor_names):
    roc_df, auc, comparator = calculate_roc(all_metrics_flat["Labels-Value"],
                                   all_metrics_flat[predictor_name], true_value="True")
    comparator = comparator.replace(">=","≥").replace("<=","≤")
    youden = roc_df["specificity"] + roc_df["sensitivity"] -1
    max_youden_loc = youden.argmax()
    max_youden = youden[max_youden_loc]
    color = px.colors.qualitative.Safe[ind]
    print(f"""At max youden ({max_youden:.1f}), thresh: {comparator}{roc_df['threshold'][max_youden_loc]:.1%}, ss:  {roc_df['sensitivity'][max_youden_loc]:.1%}, sp: {roc_df['specificity'][max_youden_loc]:.1%}""")
    fig.add_trace(go.Scatter(
        x=roc_df["specificity"], y=roc_df["sensitivity"],
        mode="lines+markers", opacity=0.75,
        name=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}",
        marker=dict(color=color),
    ))
    fig.add_annotation(
        x=roc_df["specificity"][max_youden_loc], y=roc_df["sensitivity"][max_youden_loc],
        text=f"At {comparator}{roc_df['threshold'][max_youden_loc]:.1%}",
        showarrow=True,
        font=dict(color=color),
        opacity=0.8,
        bgcolor="#FFF", bordercolor="#000",
        arrowwidth=2,
        arrowhead=1)


customize_roc_curve(fig)
fig.show()
save_plotly_figure(fig, title)


# In[353]:


np.round(roc_df["threshold"],2)


# In[352]:


formula = "{}"
formula.format(1), formula.format(2.0), formula.format(3.1), formula.format(3.14159265358), formula.format(10.1)


# In[380]:


title = "Loc ROC Curve - xy (rank)"
predictor_names = ["Ranks-Location-Mean-x", "Ranks-Location-Mean-y",
                   "Ranks-Location-SD-x", "Ranks-Location-SD-y"]

fig = go.Figure()
# Add diagonal random chance reference line
fig.add_shape(type="line", line=dict(dash="dot",width=2), x0=1, y0=0, x1=0, y1=1)

for ind, predictor_name in enumerate(predictor_names):
    roc_df, auc, comparator = calculate_roc(all_metrics_flat["Labels-Value"],
                                   all_metrics_flat[predictor_name], true_value="True")
    comparator = comparator.replace(">=","≥").replace("<=","≤")
    youden = roc_df["specificity"] + roc_df["sensitivity"] -1
    max_youden_loc = youden.argmax()
    color = px.colors.qualitative.Safe[ind]
    fig.add_trace(go.Scatter(
        x=roc_df["specificity"], y=roc_df["sensitivity"],
        mode="lines+markers+text", opacity=0.75,
        #text=roc_df["threshold"].apply((comparator+"{:}").format), textposition="bottom right",
        name=f"{var_labels[predictor_name]}, AUC: {auc:0.3f}",
        marker=dict(color=color),
        customdata=np.stack((np.round(roc_df["threshold"],2),roc_df["threshold"]),axis=-1),
        hovertemplate =
        "<b>Specificity</b>: %{x:.2%}" + "<br>" +
        "<b>Sensitivity</b>: %{y:.2%}" + "<br>" +
        "<b>Threshold</b>: " + comparator + "%{customdata[0]}"
    ))


    fig.add_annotation(
        x=roc_df["specificity"][max_youden_loc], y=roc_df["sensitivity"][max_youden_loc],
        text=f"At {comparator}{roc_df['threshold'][max_youden_loc]}",
        showarrow=True,
        font=dict(color=color),
        #arrowcolor=color,
        bgcolor="#FFF", bordercolor="#000",
        opacity=0.8,
        arrowwidth=2,
        arrowhead=2)

customize_roc_curve(fig, add_reference_line=False)
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
save_plotly_figure(fig, title)


# In[ ]:




