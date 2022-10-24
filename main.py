import pandas as pd
from IPython.display import Image
import cv2
import numpy as np
import imutils

df = pd.read_excel('data/data.xlsx')

class Image:
    img = None
    center = None
    def __init__(self, filename:str):
        index = int(filename.split('.jpg')[0].split('/')[1])
        self.img = cv2.imread(filename, cv2.IMREAD_COLOR)
        self.center = (df['centerX'][index], df['centerY'][index])

def rotate_img(img_CV2, deg:int):
    return imutils.rotate(img_CV2, deg)

def rotated_segment(img, deg:int, widthPixels:int, center:tuple):
    imgr = rotate_img(img, deg)
    mask = np.zeros(imgr.shape, dtype='uint8')
    cv2.rectangle(mask, (center[0], int(center[1] - (widthPixels / 2))), (center[0] + 25000000, int(center[1] + (widthPixels / 2))), (255,255,255), -1)
    return np.where(mask, imgr, np.zeros(imgr.shape, dtype='uint8'))

def segment_by_deg(img, degInterval:int, widthPixels:int, center:tuple):
    deg = 0
    segments = []
    for i in range(int(360 / degInterval)):
        segments.append(rotated_segment(img, deg, widthPixels, center))
        deg += degInterval
    return segments

def cropped_segments(img, degInterval:int, widthPixels:int, center:tuple):
    segments = segment_by_deg(img, degInterval, widthPixels, center)
    new_pieces = []
    for segment in segments:
        new_pieces.append(segment[int(center[1] - (widthPixels / 2)):int(center[1] + (widthPixels / 2)), int(center[0]):int(center[0] + 999999999)])
    return new_pieces

from matplotlib import pyplot as plt
def vertical_display(segments, cropped:bool, center=None, widthPixels=None):
    if cropped:
        for s in segments:
            plt.figure()
            plt.imshow(s)
    else:
        for segment in segments:
            plt.figure()
            plt.imshow(segment[int(center[1] - (widthPixels / 2)):int(center[1] + (widthPixels / 2)), int(center[0]):int(center[0] + 999999999)])

# TODO: make this work properly
def horizontal_display(segments, cropped:bool, center=None, widthPixels=None):
    # adjust/rotate images
    imgs = []
    if cropped:
        for segment in segments:
            imgs.append(rotate_img(segment, 90))
    else:
        imgs = []
        for segment in segments:
            new = segment[int(center[1] - (widthPixels / 2)):int(center[1] + (widthPixels / 2)), int(center[0]):int(center[0] + 999999999)]
            imgs.append(rotate_img(new, 90))
    # build figure and display
    plt.figure(figsize=(1, 2 * len(imgs)))
    for i in range(len(imgs)):
        plt.subplot(1, 2*len(imgs), i + 1)
        plt.imshow(imgs[i])