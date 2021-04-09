import json
import os
import shutil
import numpy as np
import utils.plot as plot
from skimage import io
from optimization.simple_gradient import generate_segmentation as simple_gradient
from optimization.refine_segmentation import (
    generate_segmentation as refine_segmentation,
)

data_dir = "data/test/"
base_dir = "testing/"

file = os.listdir(data_dir)[0]


def crop_image(img, hi=460, lo=640, le=100, ri=100, pad=0.1):
    cropped = img[hi:-lo, le:-ri, :]
    horiz = int(cropped.shape[1] * pad)
    verti = int(cropped.shape[0] * pad)
    return cropped[verti:-verti, horiz:-horiz, :]


img = io.imread(data_dir + file)
cropped = crop_image(img)


# Simple Gradient

seg_init = simple_gradient(cropped, diam_range=(60, 150))

# original
# pre: 0.26s
# denoise: 3.00s
# norm: 0.26s
# gradient: 1.54s
# tophat: 1.01s
# thresholding: 0.10s
# labels: 0.28s
# segments: 1.71s

# only Gauß denoising
# denoise: 0.53s

# simpler Segment calculation
# segments: 0.12s


# Refactoring refine_segmentation

seg = refine_segmentation(seg_init, n=2)

# original
# get_distance_matrix: 0.96s - 5.32s
# get_neighbour_distances: 0.00s - 0.00s
# get_neighbour_angles: 0.06s - 0.33s
# predict_expected_angles: 0.01s - 0.05s
# predict_new_vectors: 11.56s - 58.82s
# consolidate_vectors: 0.04s - 0.05s
# consolidate_segments: 2.79s - 5.29s

# refactored predict_new_vectors
# predict_new_vectors: 0.62s - 2.43s

# refactored consolidate_segments
# consolidate_segments: 0.55s - 1.84s

# using angles matrix for predict_new_vectors
# predict_new_vectors: 0.13s - 0.44s

plot.segmentation(cropped, seg)
