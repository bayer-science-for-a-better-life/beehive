import warnings
import numpy as np
from utils.segmentation import Segment, Segmentation, consolidate_segments
from skimage import segmentation, measure, exposure, color
from skimage.filters import threshold_otsu, rank, gaussian
from skimage.morphology import disk, white_tophat


def generate_segmentation(img, denoise_mask=3, diam_range=(150, 300), expand=1.2, max_overlap=0.5):
    '''create Segmentation object from gray-scale np image
    denoise_mask: mask used for denoising image in the beginning
    diam_range: min, max diam allowed for each cell in px; chose from min
                (just around edges of cell) to max (half into neighbor cells)
    expand: float by which to expand the region which was identified as a segment
    max_overlap: float of maximum relative overlap 2 segmentas can have
    '''
    gradient_mask = 3 if diam_range[0] > 100 else 1
    area_range = tuple(d**2 for d in diam_range)
    gray = color.rgb2gray(img)

    # denoise
    denoised = gaussian(gray, denoise_mask)

    # contrast stretching for normalization
    p2, p98 = np.percentile(denoised, (2, 98))
    normed = exposure.rescale_intensity(denoised, in_range=(p2, p98))

    # image gradient
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = rank.gradient(normed, disk(gradient_mask))

    # try to seperate gradients if very close together
    if gradient_mask < 3:
        grad = white_tophat(grad, disk(denoise_mask))

    # thresholding
    thr = threshold_otsu(grad)
    markers = np.zeros(grad.shape, dtype=float)
    markers[grad > thr] = 1

    # labels
    cleared = segmentation.clear_border(markers)
    labels = measure.label(cleared)

    # capture labeled regions in rectangles
    seg = Segmentation(img=img)
    for region in measure.regionprops(labels):
        minr, minc, maxr, maxc = region.bbox
        x = maxc - minc
        y = maxr - minr
        if min([x, y]) > diam_range[0] and max([x, y]) < diam_range[1]:
            seg.add(Segment.from_region(region=region, expand=expand))

    return consolidate_segments(seg, overlap=max_overlap)
