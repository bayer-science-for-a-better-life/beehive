import json
import os
import shutil
import time
import numpy as np
import utils.plot as plot
from skimage import io
from utils.evaluation import Validation
from simple_gradient.procedure import generate_segmentation as simple_gradient
from refine_segmentation.procedure import generate_segmentation as refine_segmentation
from multiprocessing import Pool

data_dir = "data/test/"
base_dir = "test/"
nproc = 3
write_segments = False


def crop_image(img, hi=460, lo=640, le=100, ri=100, pad=0.1):
    cropped = img[hi:-lo, le:-ri, :]
    horiz = int(cropped.shape[1] * pad)
    verti = int(cropped.shape[0] * pad)
    return cropped[verti:-verti, horiz:-horiz, :]


def reset_dir(dirpath):
    try:
        shutil.rmtree(dirpath)
    except FileNotFoundError:
        pass
    os.mkdir(dirpath)


def process(file):
    img = io.imread(data_dir + file["file_name"])
    cropped = crop_image(img)
    t0 = time.time()
    seg_init = simple_gradient(cropped, diam_range=(60, 150))
    seg = refine_segmentation(seg_init)
    t1 = time.time() - t0
    plot.segmentation(
        cropped,
        seg,
        title="%s %.2fs" % (file["file_name"], t1),
        save="%s/%s" % (savedir, file["file_name"]),
    )

    if write_segments:
        s_dir = "%s%s%s" % (base_dir, "segments/", file["name"])
        reset_dir(s_dir)
        for s_name in seg.names:
            s = seg.get(s_name, img=True)
            io.imsave("%s/%s.png" % (s_dir, s.name), s.img)


if __name__ == "__main__":
    savedir = "%ssegmentations" % base_dir
    files = []
    for f in os.listdir(data_dir):
        files.append(dict(name="BFD" + f.split(" ")[-1].split(".")[0], file_name=f))

    reset_dir(savedir)
    with Pool(nproc) as p:
        p.map(process, files)
