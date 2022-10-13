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