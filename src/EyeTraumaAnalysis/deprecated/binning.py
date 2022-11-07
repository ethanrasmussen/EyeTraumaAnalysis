import numpy as np
import cv2 as cv
import pandas as pd

image = cv.imread('data/1.jpg')

while True:
    cv.imshow('img', image)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()