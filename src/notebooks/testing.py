#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import importlib
from matplotlib import pyplot as plt

os.chdir("../..")
directory_path = os.path.abspath(os.path.join("src"))
if directory_path not in sys.path:
    sys.path.append(directory_path)

import EyeTraumaAnalysis


# In[2]:


importlib.reload(EyeTraumaAnalysis)


# In[5]:


image = EyeTraumaAnalysis.Image("data/ischemic/00001_li.jpg")
plt.imshow(image.img)


# In[6]:


plt.imshow(EyeTraumaAnalysis.rotate_img(image.img, 77))


# In[8]:


plt.imshow(EyeTraumaAnalysis.rotated_segment(img=image.img, deg=218, wd_px=20, center=image.center))


# In[11]:


pieces = EyeTraumaAnalysis.segment_by_deg(img=image.img, interval_deg=10, wd_px=20, center=image.center)
type(pieces)


# In[12]:


for piece in pieces:
    plt.figure()
    plt.imshow(piece)


# In[13]:


cropped = EyeTraumaAnalysis.cropped_segments(img=image.img[...,[2,1,0]], interval_deg=10, width_px=20, center=image
                                             .center)
type(cropped)


# In[90]:


plt.imshow(cropped[0])


# In[91]:


plt.imshow(cropped[1])


# In[92]:


EyeTraumaAnalysis.vertical_display(cropped, True)


# In[93]:


EyeTraumaAnalysis.horizontal_display(cropped, True)


# In[94]:


plt.imshow(EyeTraumaAnalysis.recenter_img(image.img, image.center))


# In[ ]:


EyeTraumaAnalysis.show_canny(image.img)


# In[ ]:


# plt.imshow(image.img[:,:,::-1])
import cv2
from PIL import Image as pilImg
import numpy as np
# plt.imshow(cv2.cvtColor(image.img, cv2.COLOR_BGR2RBG))
# b, g, r = pilImg.open('data/00001.jpg').split()
# plt.imshow(pilImg.merge("RGB", (b, r, g)))

im = pilImg.open('data/00001.jpg')
r,g,b,a = np.array(im.convert("RGBA")).T
arr = [b,r,g,a]
im = pilImg.fromarray(np.array(arr).transpose())
plt.imshow(im)


# In[ ]:


#plt.imshow(image.img[...,::-1])
plt.imshow(255-image.img[...,::-1])


# In[ ]:


im = cv2.imread('data/00001.jpg')
plt.imshow(cv2.Canny(im, 25, 100))


# In[ ]:


plt.imshow(cv2.Canny(cropped[0], 25, 100, 3, True))


# In[ ]:


plt.imshow(cropped[0])


# In[ ]:


# plt.imshow(np.mean(cropped[0], axis=0))
# np.mean(cropped[0], axis=0).shape
# plt.imshow(np.mean(cropped[0], axis=0))
averaged = np.mean(cropped[0], axis=0)
plt.imshow(averaged)


# In[95]:


plt.imshow(np.vstack(cropped))
# plt.imshow(np.mean((np.vstack(cropped)), axis=0))


# In[ ]:


cropped[0].shape


# In[ ]:


import numpy as np
plt.imshow(
    np.vstack([
        np.hstack([
            image.img[...,[0,1,2]],
            image.img[...,[1,0,2]],
            image.img[...,[2,1,0]],
            image.img[...,[0,2,1]],]),
        np.hstack([
            255-image.img[...,[0,1,2]],
            255-image.img[...,[1,0,2]],
            255-image.img[...,[2,1,0]],
            255-image.img[...,[0,2,1]],
        ])])
)

