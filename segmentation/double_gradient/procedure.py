import copy
import warnings
import numpy as np
import utils.image as image
from utils.segmentation import Segment, Segmentation, segment_overlap
from skimage import segmentation, measure, exposure, color
from skimage.filters import threshold_otsu, rank, gaussian
from skimage.morphology import disk, skeletonize


def region2segment(region, img, expand=1.2):
    '''Get segment from region based on convex hull
    region: region from regionprops()
    img: image from which region was extracted
    expand: factor by which to expand the bounding box
    '''
    minr, minc, maxr, maxc = region.bbox
    chull = image.mask_image(img[minr:maxr, minc:maxc], region.convex_image)
    cent = image.centroid(chull)
    cent = (cent[0] + minr, cent[1] + minc)
    s = Segment(
        centroid=cent,
        dx_r=int((maxc - cent[1]) * expand),
        dx_l=int((cent[1] - minc) * expand),
        dy_u=int((cent[0] - minr) * expand),
        dy_d=int((maxr - cent[0]) * expand))
    return s


def segment_fits(s, max_ar=1.5, area_range=(1000, 100000), diam_range=(150, 500)):
    '''Does segment fit into defined space?
    max_ar: maximum aspect ratio allowed
    area_range: min and max area allowed
    diam_range: min and max diam allwoed
    '''
    return s.ar_max < max_ar and \
        s.area > area_range[0] and \
        s.area < area_range[1] and \
        min([s.xdiam, s.ydiam]) > diam_range[0] and \
        max([s.xdiam, s.ydiam]) < diam_range[1]


def remove_duplicates(seg, overlap=0.7):
    '''Consolidate segments which overlap a lot. The largest segment is chosen.
    overlap: min relative overlap that defines whether 2 segments are consolidated
    '''
    segment_grps = []
    unused_segments = copy.deepcopy(seg)
    for s1 in seg:
        if s1.name not in unused_segments.names:
            continue
        unused_segments.remove(s1)
        similars = [s2 for s2 in unused_segments if segment_overlap(s1, s2) > overlap]
        segment_grps.append([s1] + similars)
        for sim in similars:
            unused_segments.remove(sim)

    seg_cons = Segmentation(seg.img)
    for grp in segment_grps:
        max_area = max([d.area for d in grp])
        seg_cons.add([d for d in grp if d.area == max_area][0])
    return seg_cons


def generate_segmentation(img, denoise_mask=3, gradient_mask=3):
    '''Create Segmentation object from color np image
    denoise_mask: mask used for denoising image in the beginning
    gradient_mask: mask used for computing gradient
    '''
    # gray scale
    gray = color.rgb2gray(img)

    # denoise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denoised = gaussian(rank.median(gray, disk(denoise_mask)), denoise_mask)

    # contrast stretching for normalization
    p2, p98 = np.percentile(denoised, (2, 98))
    normed = exposure.rescale_intensity(denoised, in_range=(p2, p98))

    # image gradient
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = rank.gradient(normed, disk(gradient_mask))

    # thresholding
    thr_h = threshold_otsu(grad)
    thr_l = threshold_otsu(grad[grad < thr_h])

    markers_h = np.zeros(grad.shape, dtype=float)
    markers_h[grad > thr_h] = 1

    markers_l = np.zeros(grad.shape, dtype=float)
    markers_l[(grad < thr_h) & (grad > thr_l)] = 1

    markers = skeletonize(markers_h) + skeletonize(markers_l)

    # labels
    cleared = segmentation.clear_border(markers)
    labels = measure.label(cleared)

    # capture labeled regions in rectangles
    seg = Segmentation(img)
    for region in measure.regionprops(labels):
        s = region2segment(region, labels)
        if segment_fits(s):
            seg.add(s)

    # remove multi-captured segments
    seg_filtered = remove_duplicates(seg)

    return seg_filtered
