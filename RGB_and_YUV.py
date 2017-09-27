
# coding: utf-8

# # Prologue: RGB and YUV
# Utility functions to convert between RGB and YUV

# In[1]:

import numpy as np

# Got the conversion matrix from here
# http://www.equasys.de/colorconversion.html
def conv_ycbcr(im):
    y = np.array([0.213, 0.715, 0.072])
    u = np.array([-0.115, -0.385, 0.5])
    v = np.array([0.5, -0.454, -0.046])
    color_matrix = np.array([y, u, v])
    yuv = im.dot(color_matrix.T)
    yuv[:,:,[1,2]] += 128
    return yuv

# Got the conversion matrix from here
# http://www.equasys.de/colorconversion.html
def conv_rgb(im):
    im[:,:,[1,2]] -= 128
    
    r = np.array([1, 0, 1.575])
    g = np.array([1, -0.187, -0.468])
    b = np.array([1, 1.856, 0])
    color_matrix = np.array([r,g,b])
    
    rgb = im.dot(color_matrix.T)
    rgb[rgb < 0] = 0
    rgb[rgb > 255] = 255
    rgb = np.uint8(rgb)
    return rgb


# In[ ]:



