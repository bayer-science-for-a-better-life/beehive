import torchvision.transforms as trans
import numpy as np
from PIL import Image
from torchvision.transforms import Compose

resize = trans.Resize((50, 50))
totensor = trans.ToTensor()


class FileImgs(object):

    def __init__(self, img_paths):
        imgs = []
        for path in img_paths:
            imgs.append(Image.open(path))
        self.imgs = imgs
        self.transform = Compose([resize, totensor])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return self.transform(self.imgs[idx])


class SegmentImgs(object):

    def __init__(self, segments):
        imgs = []
        for idx in range(len(segments)):
            seg = segments[idx]
            arr = segments.get_img(seg)
            imgs.append(Image.fromarray(np.uint8(arr)))
        self.imgs = imgs
        self.transform = Compose([resize, totensor])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return self.transform(self.imgs[idx])
