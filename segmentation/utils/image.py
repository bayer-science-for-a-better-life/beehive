import numpy as np


def centroid(img):
    """Get centroid from 2-dim image as (row, col)"""
    crow = (img.sum(axis=1) / img.sum() * np.arange(img.shape[0])).sum()
    ccol = (img.sum(axis=0) / img.sum() * np.arange(img.shape[1])).sum()
    return (int(crow), int(ccol))


def mask_image(img, mask):
    """Get binary image from bool mask and image"""
    masked = img.copy()
    masked[:, :] = 0
    masked[mask] = 1
    return masked
