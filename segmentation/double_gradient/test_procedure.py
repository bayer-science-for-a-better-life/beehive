import json
import numpy as np
import utils.plot as plot
from skimage import io, color
from utils.evaluation import Validation
from double_gradient.procedure import generate_segmentation


image_names = ['angled', 'capped', 'dark', 'egg', 'empty', 'medium', 'small']
data_dir = 'data/broodmapper/'
base_dir = 'double_gradient/'


# load original images and do segmentation
images = dict()
for image_name in image_names:
    img = io.imread(data_dir + image_name + '.jpg')
    seg = generate_segmentation(img)
    images[image_name] = dict(img=img, seg=seg)


# validate segmentation with labels
for image_name in images:
    labels_path = data_dir + image_name + '_labels.png'
    val = Validation(images[image_name]['img'], labels_path)
    val.confuse(images[image_name]['seg'])
    images[image_name]['val'] = val
    print(image_name, val)
# angled <Validation TPR 0.96 PPV 0.93/>
# capped <Validation TPR 0.60 PPV 0.89/>
# dark <Validation TPR 0.62 PPV 0.86/>
# egg <Validation TPR 0.99 PPV 0.95/>
# empty <Validation TPR 1.00 PPV 0.93/>
# medium <Validation TPR 0.96 PPV 0.95/>
# small <Validation TPR 0.99 PPV 0.83/>


# see segmentations
for image_name in images:
    seg = images[image_name]['seg']
    val = images[image_name]['val']
    plot.segmentation(
        seg.img, seg,
        title='%s TPR %.2f PPV %.2f' % (image_name, val.TPR, val.PPV),
        save='%ssegmentations/%s.png' % (base_dir, image_name))

# write results to file
recs = [dict(img=k, TPR=d['val'].TPR, PPV=d['val'].PPV) for k, d in images.items()]
with open(base_dir + 'results.json', 'w') as ouf:
    json.dump(recs, ouf)
