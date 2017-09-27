
# coding: utf-8

# In[5]:

import numpy as np

def dc_pred(pix, x, y, block_size):
    top = np.sum(pix[y-1:y, x:x+block_size]) if y > 1 else block_size * 128
    left = np.sum(pix[y:y + block_size, x-1:x]) if x > 1 else block_size * 128
    return (top + left + block_size) // (block_size * 2)


# In[6]:

def error(im1, im2):
    diff = im1 - im2
    return np.round(np.sum(np.multiply(diff,diff)))


# In[7]:

def scale(ac_con, alpha_pos, alpha_neg, block_size):
    scaled = np.zeros((block_size, block_size))
    mask = ac_con < 0
    scaled[mask] = ac_con[mask] * alpha_neg 
    mask = ~mask
    scaled[mask] = ac_con[mask] * alpha_pos
    return scaled


# In[10]:

def pick_alpha(ac_con, mags, block_size, dc_pred_u, dc_pred_v, block_u, block_v):
    min_err_u = 10000000000
    min_err_v = 10000000000
    for alpha in mags:
        scaled = ac_con * alpha
        scaled_u = dc_pred_u + scaled
        err = error(block_u, scaled_u)
        if err < min_err_u:
            min_err_u = err
            best_a_u = alpha
        scaled_v = dc_pred_v + scaled
        err = error(block_v, scaled_v)
        if err < min_err_v:
            min_err_v = err    
            best_a_v = alpha           
    return best_a_u, best_a_v


# In[8]:

def pick_alpha_pos_neg(ac_con, mags, block_size, dc_pred_u, dc_pred_v, block_u, block_v):
    min_err_u = 10000000000
    min_err_v = 10000000000
    
    for alpha_pos in mags:
        for alpha_neg in mags:
            scaled_u = scale(ac_con, alpha_pos, alpha_neg, block_size) + dc_pred_u
            err = error(block_u, scaled_u)
            if err < min_err_u:
                min_err_u = err
                best_a_pos_u = alpha_pos
                best_a_neg_u = alpha_neg
            
            scaled_v = scale(ac_con, alpha_pos, alpha_neg, block_size) + dc_pred_v
            err = error(block_v, scaled_v)
            if err < min_err_v:
                min_err_v = err
                best_a_pos_v = alpha_pos
                best_a_neg_v = alpha_neg
    return best_a_pos_u, best_a_neg_u, best_a_pos_v, best_a_neg_v


# In[11]:

def psnr(im1, im2):
    h, w = im1.shape
    sse = np.sum((im1 - im2)**2)
    return 20 * np.log10(255) - 10 * np.log10(sse/(h*w))


# In[ ]:



