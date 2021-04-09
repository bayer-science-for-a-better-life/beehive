'''Feature Preparation

This NN should be able to classify the major classes.
The ones which were not massively underrepresented.
Underrepresented classes were:
- unknown: label 10
- hatching: label 6
- dead: label 9
Major classes: 0, 1, 2, 3, 4, 5, 7, 8

For the NN I have to relabel them from 0 to 7.
In another labels.json I save the mapping.

In addition I will do a train-val-split (2:1) and write everything into
`major_classes` LMDB with `train` and `val` sub-DBs.
'''
import random
import json
import pandas as pd
import utils.plots as plots
from utils.dbconnector import DBConnector

major_labels = [0, 1, 2, 3, 4, 5, 7, 8]


# relabeling
with open('label_maps.json', 'rb') as inf:
    label_maps = json.load(inf)

label_maps = [d for d in label_maps if d['label'] in major_labels]
labels = []
for new_label, map in enumerate(label_maps):
    labels.append(dict(
        name='%d %s' % (new_label, map['name'].split(' ')[1]),
        label=new_label,
        old_label=map['label']))

with open('major_classes/labels.json', 'w') as ouf:
    json.dump(labels, ouf)


# label map for translating from old to new label
label_map = {d['old_label']: d['label'] for d in labels}


# get idxs which have datapoints for major classes
db = DBConnector('dbs/master')
db.open()
labels = db.get_all_targets('train')
db.close()

keep = [d in major_labels for d in labels]
keep_idxs = [i for i, d in enumerate(keep) if d]
sum(keep) / len(keep)  # 0.96


# create partitions
parts = random.choices(['train', 'train', 'val'], k=len(keep_idxs))
sum(d == 'train' for d in parts)  # 3461
sum(d == 'val' for d in parts)  # 1691


# write major classes DB
db_master = DBConnector('dbs/master')
db_major = DBConnector('dbs/major_classes')
db_major.clear_db()

db_master.open()
db_major.open()
for idx, part in zip(keep_idxs, parts):
    key = db_major.get_next_key(part)
    imgs, labels_old = db_master.get_values([idx], 'train')
    label_new = label_map[labels_old[0]]
    db_major.write_data(key, (imgs[0], label_new), part)
db_master.close()
db_major.close()


# sanity check
db_major.open()
db_major.get_next_key('train')  # 3461
db_major.get_next_key('val')  # 1691
labels_val = db_major.get_all_targets('val')
labels_train = db_major.get_all_targets('train')
db_major.close()


# see label distributions
df = pd.DataFrame(dict(
    label=labels_val + labels_train,
    part=['val'] * len(labels_val) + ['train'] * len(labels_train))) \
    .groupby(['label', 'part']) \
    .size() \
    .to_frame('count') \
    .reset_index()

plots.barplot(df, dict(x='label', y='count', hue='part'), dodge=True,
              title='Label Distributions for Major Classes',
              save='major_classes/plots/label_distributions.png')
