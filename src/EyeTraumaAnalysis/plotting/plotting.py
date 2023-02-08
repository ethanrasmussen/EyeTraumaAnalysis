import plotly.express as px
import os
import numpy as np
import pandas as pd

notebook_filename = "notebook.ipynb"

#@markdown ### func `def get_path_to_save(...):`
def get_path_to_save(plot_props:dict=None, file_prefix="", save_filename:str=None, save_in_subfolder:str=None, extension="jpg", dot=".", create_folder_if_necessary=True):
    """
    Code created myself (Rahul Yerrabelli)
    """
    replace_characters = {
        "$": "",
        "\\frac":"",
        "\\mathrm":"",
        "\\left(":"(",
        "\\right)":")",
        "\\left[":"[",
        "\\right]":"]",
        "\\": "",
        "/":"-",
        "{": "(",
        "}": ")",
        "<":"",
        ">":"",
        "?":"",
        "_":"",
        "^":"",
        "*":"",
        "!":"",
        ":":"-",
        "|":"-",
        ".":"_",
    }

    # define save_filename based on plot_props
    if save_filename is None:
        save_filename = "unnamed"

    save_path = [
                 "outputs",
                f"{notebook_filename.split('.',1)[0]}",
                ]
    if save_in_subfolder is not None:
        if isinstance(save_in_subfolder, (list, tuple, set, np.ndarray) ):
            save_path.append(**save_in_subfolder)
        else:  # should be a string then
            save_path.append(save_in_subfolder)
    save_path = os.path.join(*save_path)

    if not os.path.exists(save_path) and create_folder_if_necessary:
        os.makedirs(save_path)
    return os.path.join(save_path, file_prefix+save_filename+dot+extension)
    #plt.savefig(os.path.join(save_path, save_filename+dot+extension))


default_plotly_save_scale = 4

def save_plotly_figure(fig, file_name:str, animated=False, scale=None, save_in_subfolder:str=None, extensions=None):
    """
    - for saving plotly.express figures only - not for matplotlib
    - fig is of type plotly.graph_objs._figure.Figure,
    - Requires kaleido installation for the static (non-animated) images
    """
    if scale is None:
        scale = default_plotly_save_scale
    if extensions is None:
        extensions = ["html"]
        if not animated:
            # options = ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json']
            extensions += ["eps","png","pdf"]

    for extension in extensions:
        try:
            if extension in ["htm","html"]:
                #fig.update_layout(title=dict(visible=False))
                fig.write_html( get_path_to_save(save_filename=file_name, save_in_subfolder=save_in_subfolder, extension=extension),
                    full_html=False,
                    include_plotlyjs="directory" )
            else:
                #if extension == "png":
                #    fig.update_layout(title=dict(visible=False))
                fig.write_image(get_path_to_save(save_filename=file_name, save_in_subfolder=save_in_subfolder, extension=extension), scale=scale)
        except ValueError as exc:
            import traceback
            #traceback.print_exception()

#col_options = {col_name:pd.unique(df_long[col_name]).tolist() for col_name in consistent_cols}
#display(col_options)

def customize_figure(fig, width=640, height=360, by_mmHg=True, br_ct=1, space_ct=1, textposition="inside", textfont_color=None) -> dict:
    """ - for plotly figures only. """
    # Plotly must be version 5.7.0+ for ticklabelstep to work
    if by_mmHg:
        fig.update_xaxes( #tickprefix="At ",   # Dr. WJ and Ashkhan didn't like it
                         ticksuffix="mmHg", showtickprefix="all", showticksuffix="all", tickfont=dict(size=16),
                        mirror=True, linewidth=2,
                        title=dict(text="<b>Applied Circumferential Pressure</b>", font=dict(size=20, family="Arial Black")),
                        )
        fig.update_yaxes(tickformat=".0%", tickwidth=2,  nticks=21, ticklabelstep=4,
                        mirror="ticks", linewidth=2, range=(0,1),
                        title=dict(text="<b>Obstruction of<br>Field of View (S.E.)</b>",font=dict(size=18, family="Arial Black")),
                        #title=dict(text="Width Obstructed of<br>Field of View (S.E.)",font=dict(size=18, family="Arial Black")),
                        showgrid=True, gridcolor="#DDD",
                        showspikes=True, spikemode="across", spikethickness=2, spikedash="solid", # ticklabelposition="inside top",
                        )
    #fig.update_traces(textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        font=dict(
            family="Arial",
            size=16,
            color="black",
        ),
        title={
            "y":1,
            "x":0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font":dict(size=16)
        },
        width=width, height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            title={"font_family": "Arial Black",},
            yanchor="middle",
            y=0.5,
            xanchor="center",
            x=0.08,
            #bgcolor="LightSteelBlue",
            bordercolor="Black", #font_size=16,
            borderwidth=2,
        ),
        bargap=0.05, bargroupgap=0.0,
        dragmode="drawopenpath",
        newshape_line_color="cyan",
    )

    if textfont_color is None:
        if isinstance(textposition, (list, tuple, set, np.ndarray, pd.Series) ):
            textfont_color = ["#FFF" if textposition_each == "inside" else "#000" for textposition_each in textposition]
            print(textfont_color)
        elif textposition == "inside":
            textfont_color="#FFF"
        else:
            textfont_color="#000"
    try:
        fig.update_traces(textfont_size=16, textangle=0, textfont_color=textfont_color,
                          textposition=textposition, cliponaxis=False, #textfont_family="Courier",
                          marker_line_color="#000", marker_line_width=2
                        )
    except ValueError:
        fig.update_traces(textposition=textposition, cliponaxis=False,  # textfont_family="Courier",
                          marker_line_color="#000", marker_line_width=2
                          )

    if by_mmHg:
        if textposition == "inside":
            fig.update_traces(texttemplate=[None]+[("&nbsp;"*space_ct)+("<br>"*br_ct)+"<b>%{y:.1%}</b>"]*5,)
        else:
            fig.update_traces(texttemplate=[None]+["<b>%{y:.1%}</b>"+("<br>"*br_ct)+("&nbsp;"*space_ct)]*5,)


    config = {
        "toImageButtonOptions" : {
            "format": "png", # one of png, svg, jpeg, webp
            "filename": 'custom_image',
            "scale": default_plotly_save_scale # Multiply title/legend/axis/canvas sizes by this factor
        },
        "modeBarButtonsToAdd": ["drawline","drawopenpath","drawclosedpath","drawcircle","drawrect","eraseshape"]
    }

    return config