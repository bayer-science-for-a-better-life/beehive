import warnings
import numpy as np
from utils.segmentation import Segment, Segmentation, consolidate_segments
from skimage import segmentation, measure, color
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
from skimage.util import img_as_ubyte


def generate_segmentation(img, denoise_mask=3, diam_range=(150, 300), max_overlap=0.5):
    '''create Segmentation object from gray-scale np image
    denoise_mask: mask used for denoising image in the beginning
    diam_range: min, max diam allowed for each cell in px; chose from min
                (just around edges of cell) to max (half into neighbor cells)
    max_overlap: float of maximum relative overlap 2 segmentas can have
    '''
    area_range = tuple(d**2 for d in diam_range)

    # gray scale
    gray = color.rgb2gray(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gray = img_as_ubyte(gray)

    # denoise
    denoised = rank.median(gray, disk(denoise_mask))

    # local histogram equalization
    normed = rank.equalize(denoised, selem=disk(diam_range[0]))

    # thresholding
    thr = threshold_otsu(normed)
    markers = np.zeros(normed.shape, dtype=float)
    markers[normed < thr] = 1

    # refine
    filled = binary_fill_holes(markers)
    cleared = segmentation.clear_border(filled)

    # get labels
    labels = measure.label(cleared)

    # capture labeled regions in rectangles
    seg = Segmentation(img)
    for region in measure.regionprops(labels):
        s = Segment.from_region(region, labels)
        if s.fits(max_ar=1.5, diam_range=diam_range, area_range=area_range):
            seg.add(s)

    return consolidate_segments(seg, overlap=max_overlap)
