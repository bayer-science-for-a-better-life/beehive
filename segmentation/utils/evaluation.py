import copy
import numpy as np
import utils.image as image
from skimage import io, color, measure
from utils.segmentation import Segment, Segmentation


class Validation(object):
    def __init__(self, img, labels_path):
        self.img = img
        self.TPR = None
        self.labels = self.read_labels_file(labels_path)
        self.truth = self.create_true_segmentation()

    def read_labels_file(self, path):
        """Read w/ Gimp prepared labels-file as binary image"""
        img = io.imread(path)
        gray = color.rgb2gray(img)
        th = (gray.max() - gray.min()) / 2
        gray[gray < th] = 0
        gray[gray >= th] = 1
        return measure.label(gray)

    def create_true_segmentation(self):
        segments = Segmentation(img=self.img)
        for region in measure.regionprops(self.labels):
            segments.add(Segment.from_region(region, 1.2))
        return segments

    def confuse(self, segments):
        TPs = []
        FNs = list(range(len(self.truth)))
        FPs = list(range(len(segments)))

        # check if true segments are within predicted segments
        for idx_true in range(len(self.truth)):
            seg_true = self.truth[idx_true]
            captured = False
            for idx_pred in range(len(segments)):
                seg_pred = segments[idx_pred]
                if (
                    seg_pred.xrange[0] <= seg_true.xrange[0]
                    and seg_pred.xrange[1] >= seg_true.xrange[1]
                    and seg_pred.yrange[0] <= seg_true.yrange[0]
                    and seg_pred.yrange[1] >= seg_true.yrange[1]
                ):
                    captured = True
                    break
            if captured:
                FPs.remove(idx_pred)
                TPs.append(idx_true)
                FNs.remove(idx_true)
        self.TPs = TPs
        self.FNs = FNs
        self.FPs = FPs
        nTPs = len(self.TPs)
        nFNs = len(self.FNs)
        nFPs = len(self.FPs)
        self.TPR = nTPs / (nTPs + nFNs) if nTPs + nFNs > 0 else np.nan
        self.PPV = nTPs / (nTPs + nFPs) if nTPs + nFPs > 0 else np.nan

    def __repr__(self):
        info = "" if self.TPR is None else "TPR %.2f PPV %.2f" % (self.TPR, self.PPV)
        return "<Validation %s/>" % info
