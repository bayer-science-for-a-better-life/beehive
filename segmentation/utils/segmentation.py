import random
import numpy as np


def consolidate_segments(segments, overlap):
    """Consolidate segmentation by relative overlap of segments
    A new segmentation is returned in which all `segments` of the original
    segmentation that have a relative overlap higher than `overlap` were
    averaged to a single segment each.
    relative overlap = intersection area / union area
    """
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
            out.add(Segment(xrange=xranges.mean(axis=0), yrange=yranges.mean(axis=0)))

    return out


class Segment(object):
    """
    centroid: (row, col) of centroid
    dx_r, dx_l: #pixels/cols to right/left boundary of segment
    dy_u, dy_d: #pixels/rows to upwards/downwards boundary of segment
    ranges are [range[0]; range[1])
    max aspect ratio is always `ar_max` >= 1
    """

    def __init__(self, xrange, yrange, img=None):
        self.xrange = tuple(int(d) for d in xrange)
        self.yrange = tuple(int(d) for d in yrange)
        self.img = img

    def __repr__(self):
        self.calculate_properties()
        return "<Segment %s: AR %.1f Area %d />" % (self.name, self.ar, self.area)

    def get_centroid(self):
        """get centroid (row, col)"""
        return (int(np.mean(self.yrange)), int(np.mean(self.xrange)))

    def get_diams(self):
        """get segment diameters (xdiam, ydiam)"""
        return self.xrange[1] - self.xrange[0], self.yrange[1] - self.yrange[0]

    def get_area(self):
        """get segment area"""
        return self.xdiam * self.ydiam

    def calculate_properties(self):
        self.centroid = self.get_centroid()
        self.name = "%d-%d_x%d" % (*self.centroid, random.getrandbits(32))
        self.xdiam, self.ydiam = self.get_diams()
        self.ar = self.xdiam / self.ydiam
        self.ar_max = max([self.xdiam, self.ydiam]) / min([self.xdiam, self.ydiam])
        self.area = self.get_area()

    @staticmethod
    def from_vector(x, y, diam):
        """get segment from diameter and vector pointing to centroid"""
        pad = diam / 2
        return Segment([x - pad, x + pad], [y - pad, y + pad])

    @staticmethod
    def from_region(region, expand=1.2):
        """Get segment from region based on convex hull
        region: region from regionprops()
        expand: factor by which to expand the bounding box
        """
        minr, minc, maxr, maxc = region.bbox
        factor = (expand - 1) / 2
        xpad = (maxc - minc) * factor
        ypad = (maxr - minr) * factor
        return Segment(
            xrange=[minc - xpad, maxc + xpad], yrange=[minr - ypad, maxr + ypad]
        )


class Segmentation(object):
    """Combines collection of Segments"""

    def __init__(self, segments=None, img=None):
        if segments is None:
            self.segments = []
        else:
            self.segments = segments
        self.img = img

    def __getitem__(self, idx):
        return self.segments[idx]

    def __len__(self):
        return len(self.segments)

    def __repr__(self):
        ars = []
        diams = []
        areas = []
        for d in self.segments:
            d.calculate_properties()
            ars.append(d.ar)
            diams.append(d.xdiam)
            diams.append(d.ydiam)
            areas.append(d.area)
        info = ""
        if len(self) > 0:
            info = (
                "\n\tMin\tMean\tMed\tMax"
                + "\n Diams\t%d\t%.1f\t%.1f\t%d"
                % (min(diams), np.mean(diams), np.median(diams), max(diams))
                + "\n ARs\t%.1f\t%.1f\t%.1f\t%.1f"
                % (min(ars), np.mean(ars), np.median(ars), max(ars))
                + "\n Areas\t%d\t%.1f\t%.1f\t%d"
                % (min(areas), np.mean(areas), np.median(areas), max(areas))
            )
        return "<Segmentation %d segments" % len(self) + info + " />"

    def add(self, segment):
        self.segments.append(segment)

    def get_img(self, segment):
        """crop img for segment"""
        return self.img[
            segment.yrange[0] : segment.yrange[1], segment.xrange[0] : segment.xrange[1]
        ]

    def get_overlap_matrix(self):
        """Get distance matrix of segments
        where the distance is the relative overlap of 2 segmments
        triu form: [[1, d, d],
                     0, 1, d],
                     0, 0, 1]]
        """
        n = len(self)
        xranges = np.ndarray((n, 2))
        yranges = np.ndarray((n, 2))
        for i in range(n):
            s = self[i]
            xranges[i] = s.xrange
            yranges[i] = s.yrange

        X0 = np.stack(np.meshgrid(xranges[:, 0], xranges[:, 0]), axis=2)
        X1 = np.stack(np.meshgrid(xranges[:, 1], xranges[:, 1]), axis=2)
        Y0 = np.stack(np.meshgrid(yranges[:, 0], yranges[:, 0]), axis=2)
        Y1 = np.stack(np.meshgrid(yranges[:, 1], yranges[:, 1]), axis=2)
        A = (xranges[:, [1]] - xranges[:, [0]]) * (yranges[:, [1]] - yranges[:, [0]])

        Xin = np.min(X1, axis=-1) - np.max(X0, axis=-1)
        Yin = np.min(Y1, axis=-1) - np.max(Y0, axis=-1)
        Xin[Xin < 0] = 0
        Yin[Yin < 0] = 0
        In = Xin * Yin
        Un = A.T + A - In
        return np.triu(In / Un)
