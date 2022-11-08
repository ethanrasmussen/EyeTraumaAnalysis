import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import imutils


# import pupil center data for images
li_df = pd.read_excel("data/ischemic/data_li.xlsx")
h_df = pd.read_excel("data/healthy/data_h.xlsx")
photos_df = pd.read_excel("data/photo_files_data.xlsx")


#############################################################################
################# CLASS DEFINITIONS:
#############################################################################

# special class for holding cv2 images & their pupil center points in one object
class Image:
    img = None
    center = None

    def __init__(self, filename: str):
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


# returns image rotated by deg degrees
def rotate_img(img_cv2, deg: int):
    return imutils.rotate(img_cv2, deg)


# rotates image, then returns segment masked from given center (with defined width/Y value)
def rotated_segment(img, deg: int, wd_px: int, center: tuple):
    imgr = rotate_img(img, deg)
    mask = np.zeros(imgr.shape, dtype="uint8")
    cv2.rectangle(mask,
                  (center[0], int(center[1] - (wd_px / 2))),
                  (center[0] + 25000000, int(center[1] + (wd_px / 2))),
                  (255,255,255), -1)
    return np.where(mask, imgr, np.zeros(imgr.shape, dtype="uint8"))


# given turning/degree interval, returns multiple rotated uncropped segments encompassing entire image
def segment_by_deg(img, interval_deg: int, wd_px: int, center: tuple):
    deg = 0
    segments = []
    for i in range(int(360 / interval_deg)):
        segments.append(rotated_segment(img, deg, wd_px, center))
        deg += interval_deg
    return segments

# TODO: fix math/function so it works properly
# given cv2 image file and pupil center, crops image to make given center actual geometric center of image


def recenter_img(img, center):
    x1,y1,x2,y2 = 0,0,0,0
    ## img_cv2 = img.img
    ## center = img.center
    ## height, width = img_cv2.shape[0], img_cv2.shape[1]
    height, width = img.shape[0], img.shape[1]
    wseg1, wseg2 = center[0], (width - center[0])
    hseg1, hseg2 = center[1], (height - center[1])
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


# captures entire image in cropped segments
def get_segments(img, interval_deg: int, wd_px: int, center: tuple):
    segments = segment_by_deg(img, interval_deg, wd_px, center)
    new_pieces = []
    for segment in segments:
        new_pieces.append(
            segment[int(center[1] - (wd_px / 2)):int(center[1] + (wd_px / 2)),
            int(center[0]):int(center[0] + 999999999)])
    return new_pieces


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
#     segments = segment_by_deg(img, interval_deg, wd_px, center)
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