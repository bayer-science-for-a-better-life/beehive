"""Sanity check for infer_bees"""
import os
from datasets import FileImgs
from infer import infer_bees

example_dir = 'example_imgs'
files = os.listdir(example_dir)
paths = [os.path.join(example_dir, d) for d in files]

ds = FileImgs(paths)
res = infer_bees(ds)

for i in range(len(ds)):
    decision = res[i]['class_caller']
    print(f'{files[i]} - {decision}')
