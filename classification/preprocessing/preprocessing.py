"""Preprocessing

According to `label_maps.json` images are loaded from various subdirs under `data/`
and inserted into a LMDB (`dbs/master/`) as `test` and `train` sub-DBs.
Train-test split is 2-1.

Then transformations for resizsing and data augmentation are tried out.
To visually check them, samples are plotted as raw, resized and several augmentations.
The transformation are copied to `utils/transforms.py`
"""
import json
import os
import random
from utils.dbconnector import DBConnector
import torchvision.transforms as trans
from PIL import Image
import matplotlib.pyplot as plt


# write Master DB

# maps labels and subdirectories
with open("label_maps.json", "rb") as inf:
    label_maps = json.load(inf)


# partition samples for a subdir
def partition_samples(subdir, data_dir="data", partitions=["train", "train", "test"]):
    files = os.listdir(os.path.join(data_dir, str(subdir)))
    all_paths = [os.path.join(data_dir, str(subdir), file) for file in files]
    all_parts = random.choices(partitions, k=len(files))
    return all_paths, all_parts


# lmdb
db = DBConnector("dbs/master")
db.clear_db()

db.open()
for label_map in label_maps:
    print("label", label_map["label"])
    for subdir in label_map["subdirs"]:
        paths, parts = partition_samples(subdir)
        for path, part in zip(paths, parts):
            key = db.get_next_key(part)
            img = Image.open(path)
            db.write_data(key, (img, label_map["label"]), part)
db.close()


# Image Augmentation

# val and test data will be resized to 150x150 px
# train will also be augmented
resize = trans.Resize((150, 150))
totensor = trans.ToTensor()

augment = trans.Compose(
    [
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        trans.RandomAffine(degrees=0.05, scale=(0.9, 1.1), shear=2),
        trans.RandomResizedCrop((150, 150), scale=(0.9, 1), ratio=(1, 1)),
        trans.RandomVerticalFlip(),
        trans.RandomHorizontalFlip(),
    ]
)


# I will visually check if image augmentation is to strong or to weak
# thus, function below
def plot_augmented_data(img, title=None, save=None):
    fig = plt.figure()
    fig.set_figwidth(10)

    ax = plt.subplot(2, 4, 1)
    plt.tight_layout()
    ax.set_title("Raw")
    plt.imshow(img)

    ax = plt.subplot(2, 4, 2)
    plt.tight_layout()
    ax.set_title("Resized")
    plt.imshow(resize(img))

    for i in range(6):
        ax = plt.subplot(2, 4, 3 + i)
        plt.tight_layout()
        ax.set_title("Augmented%d" % (i + 1))
        plt.imshow(augment(img))

    if title is not None:
        plt.subplots_adjust(top=0.8)
        fig.suptitle(title, fontsize=16)
    plt.show()
    if save is not None:
        fig.savefig(save)


# sample images and plot them raw, resized, and several augmentations
db = DBConnector("dbs/master")
db.open()
n = db.get_next_key("train")
subset = random.sample(range(n), k=20)

for sample in subset:
    feats, targets = db.get_values([sample], "train")
    idx = [d["label"] for d in label_maps].index(targets[0])
    plot_augmented_data(
        feats[0],
        title=label_maps[idx]["name"],
        save=os.path.join(
            "preprocessing",
            "previews",
            "label%d_key%d.png" % (label_maps[idx]["label"], sample),
        ),
    )
db.close()
