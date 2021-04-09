import time
import numpy as np
from utils.segmentation import Segmentation, Segment


def generate_segmentation(n):
    X = np.random.uniform(low=0, high=100, size=n)
    Y = np.random.uniform(low=0, high=100, size=n)
    seg = Segmentation()
    for i in range(n):
        seg.add(Segment.from_vector(X[i], Y[i], 6))
    return seg


def consolidate_segments(segments, overlap):
    D = segments.get_overlap_matrix()
    out = Segmentation(img=segments.img)
    rmvd = []
    for i in range(len(segments)):
        if i in rmvd:
            continue
        idxs = list(np.where(D[i] > overlap)[0])
        n = len(idxs)
        if n > 0:
            rmvd = rmvd + idxs
            xranges = np.ndarray((n, 2))
            yranges = np.ndarray((n, 2))
            for j, idx in enumerate(idxs):
                sim_seg = segments[idx]
                xranges[j] = sim_seg.xrange
                yranges[j] = sim_seg.yrange
            out.add(Segment(
                xrange=xranges.mean(axis=0),
                yrange=yranges.mean(axis=0)))

    return out


seg = generate_segmentation(5000)
t0 = time.time()
res = consolidate_segments(seg, 0.5)
print('%d -> %d: %.2fs' % (len(seg), len(res), time.time() - t0))
# 5000 -> 1359: 5.95s original
# 5000 -> 1330: 5.93s reduced Segmentation init calcs
# 5000 -> 1337: 6.23s using xrange/yrange only
# 5000 -> 1528: 4.84s using get_overlap_matrix
# 5000 -> 1503: 4.82s few small changes
