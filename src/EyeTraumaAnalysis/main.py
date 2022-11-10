import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import imutils


# Get pupil center data for images from excel
# opencv function cv2.getRotationMatrix2D gives an error if center input is np.int64 instead of python raw int or float
li_df = pd.read_excel("data/01_raw/data_li.xlsx", dtype={"centerX":float, "centerY":float})
h_df = pd.read_excel("data/01_raw/data_h.xlsx")
photos_df = pd.read_excel("data/01_raw/photo_files_data.xlsx")  # contains info on all the images


#############################################################################
################# CLASS DEFINITIONS:
#############################################################################

class Image:
    """special class for holding cv2 images & their pupil center points in one object """
    img = None
    center = None  # tuple of np.float64

    def __init__(self, filename: str=None, img=None, center: tuple=None):
        if filename is None:
            self.img = img
            self.center = center
        else:
            index = int(filename.split(".jpg")[0].split("/")[-1].replace("_h", "").replace("_li", ""))
            self.img = cv2.imread(filename, cv2.IMREAD_COLOR)[..., [2,1,0]]
            if "_li" in filename:
                self.center = (li_df["centerX"][index], li_df["centerY"][index])
            elif "_h" in filename:
                self.center = (h_df["centerX"][index], h_df["centerY"][index])
            else:
                self.center = None

#############################################################################
################# HELPER FUNCTIONS:
#############################################################################

def rotate_img(img_cv2, deg:float, center:tuple = None):
    """shortcut function to return image rotated by deg degrees"""
    if center is None:  # if the center is None (default), initialize it as the center of the image
        center = (int(img_cv2.shape[1] / 2), int(img_cv2.shape[0] / 2))
    # imutils.rotate's default center is None (converted to middle of image), default scale is 1.0
    # imutils uses opencv functions (warpAffine and getRotationMatrix2D) to rotate
    return imutils.rotate(img_cv2, deg, center=center, scale=1.0)


def get_segment_uncropped(img, deg:float, wd_px:int, center:tuple = None, side_left:bool = False):
    """rotates image, then returns segment masked from given center (with defined width/Y value)
    Function formerly named get_segment_uncropped
    """
    if center is None:  # if the center is None (default), initialize it as the center of the image
        center = (int(img.shape[1]/2), int(img.shape[0]/2))
    img_rotated = rotate_img(img, deg, center=center)
    mask = np.zeros(img_rotated.shape, dtype="uint8")
    if side_left:
        cv2.rectangle(mask,
                      (int(center[0] - 25000000), int(center[1] - (wd_px / 2))),
                      (int(center[0]), int(center[1] + (wd_px / 2))),
                      (255, 255, 255), -1)
    else:  # segments on the right side
        cv2.rectangle(mask,
                      (int(center[0]), int(center[1] - (wd_px / 2))),
                      (int(center[0] + 25000000), int(center[1] + (wd_px / 2))),
                      (255, 255, 255), -1)

    return np.where(mask, img_rotated, np.zeros(img_rotated.shape, dtype="uint8"))


def get_segments_uncropped(img, interval_deg:int, wd_px:int, center:tuple = None, \
                           side_left:bool = False, clockwise:bool = True):
    """given turning/degree interval, returns multiple rotated uncropped segments encompassing entire image
    Function formerly named
    :return:
    :rtype: list
    """
    if center is None:  # if the center is None (default), initialize it as the center of the image
        center = (img.shape[1] // 2, img.shape[0] // 2)
    segments = {}
    deg = 0 if clockwise else 360  # deg will either go from 0->360 or 360->0
    for i in range(int(360 / interval_deg)):
        segments[deg] = get_segment_uncropped(img, deg, wd_px, center=center, side_left=side_left)
        deg = deg + (interval_deg if clockwise else -interval_deg)
    return segments

# TODO: fix math/function so it works properly
# given cv2 image file and pupil center, crops image to make given center actual geometric center of image


def recenter_img(img, center:tuple = None):
    if center is None:  # if the center is None (default), initialize it as the center of the image
        center = (int(img.shape[1]/2), int(img.shape[0]/2))
    x1,y1,x2,y2 = 0,0,0,0
    ## img_cv2 = img.img
    ## center = img.center
    ## height, width = img_cv2.shape[0], img_cv2.shape[1]
    height, width = img.shape[0], img.shape[1]
    wseg1, wseg2 = int(center[0]), int((width - center[0]))
    hseg1, hseg2 = int(center[1]), int((height - center[1]))
    wdiff = max([wseg1, wseg2]) - min([wseg1, wseg2])
    if wseg1 > wseg2:
        x1 = x1 + wdiff
    if wseg2 > wseg1:
        x2 = x2 - wdiff
    hdiff = max([hseg1, hseg2]) - min([hseg1, hseg2])
    if hseg2 > hseg1:
        y2 = y2 - hdiff
    if hseg1 > hseg2:
        y1 = y1 + hdiff
    ## return img_cv2[x1:y1, x2:y2]
    return img[x1:y1, x2:y2]


#############################################################################
################# RADIAL SEGMENTATION FUNCTIONS:
#############################################################################

def get_segments(img, interval_deg:int, wd_px:int, center:tuple = None, side_left:bool = False):
    """captures entire image in cropped segments
    :return: list of images (image segments)
    :rtype: list
    """
    if center is None:  # if the center is None (default), initialize it as the center of the image
        center = (img.shape[1] // 2, img.shape[0] // 2)   # 2-length tuple of ints
    segments = get_segments_uncropped(img, interval_deg, wd_px, center=center, side_left=side_left)
    cropped_segments = {}
    for deg, segment in segments.items():
        if side_left:
            cropped_segments[deg] = segment[
                                        int(center[1] - (wd_px / 2)):int(center[1] + (wd_px / 2)),
                                        int(center[0] - 999999999):int(center[0])
                                    ]
        else:  # segments on the right side
            cropped_segments[deg] = segment[
                                        int(center[1] - (wd_px / 2)):int(center[1] + (wd_px / 2)),
                                        int(center[0]):int(center[0] + 999999999)
                                    ]
    return cropped_segments


#############################################################################
################# EDGE DETECTION CODE:
#############################################################################

# TODO: tweak Canny edge detection or find alternative edge detection method that functions well
# goal: detect limbus within individual segments
def canny_edges(image_img):
    return cv2.Canny(image_img, 100, 200)


def show_canny(img_img):
    im = canny_edges(img_img)
    plt.subplot(121), plt.imshow(img_img, cmap="gray")
    plt.title("Original Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(im, cmap="gray")
    plt.title("Edge Image"), plt.xticks([]), plt.yticks([])
    plt.show()


#############################################################################
################# DEPRECATED CODE:
#############################################################################

# def get_segments(img, interval_deg:int, wd_px:int):
#     center = (int(img.shape[1]/2), int(img.shape[0]/2))
#     segments = get_segments_uncropped(img, interval_deg, wd_px, center)
#     new_pieces = []
#     for segment in segments:
#         new_pieces.append(segment[int(center[1] - (wd_px / 2)):int(center[1] + (wd_px / 2)),
#                           int(center[0]):int(center[0] + 999999999)])
#     return new_pieces

# def vertical_display(segments, cropped:bool, center=None, wd_px=None):
#     if cropped:
#         for s in segments:
#             plt.figure()
#             plt.imshow(s)
#     else:
#         for segment in segments:
#             plt.figure()
#             plt.imshow(segment[int(center[1] - (wd_px / 2)):int(center[1] + (wd_px / 2)), int(center[0]):int(center[0] + 999999999)])
#
# def horizontal_display(segments, cropped:bool, center=None, wd_px=None):
#     # adjust/rotate images
#     imgs = []
#     if cropped:
#         for segment in segments:
#             imgs.append(rotate_img(segment, 90))
#     else:
#         imgs = []
#         for segment in segments:
#             new = segment[int(center[1] - (wd_px / 2)):int(center[1] + (wd_px / 2)), int(center[0]):int(center[0] + 999999999)]
#             imgs.append(rotate_img(new, 90))
#     # build figure and display
#     plt.figure(figsize=(1, 2 * len(imgs)))
#     for i in range(len(imgs)):
#         plt.subplot(1, 2*len(imgs), i + 1)
#         plt.imshow(imgs[i])