"""Collect labeled images
Walk through directories and copy images from all class-subdirectories
into ./all/. Will delete current ./all/ first.
"""
import os
import shutil

basedir = "test/segments/"
targetdir = basedir + "all/"
labels = [
    "empty",
    "egg",
    "small_larva",
    "medium_larva",
    "big_larva",
    "nectar",
    "pollen",
    "capped",
    "dead",
]

# reset all/
if os.path.isdir(targetdir):
    shutil.rmtree(targetdir)
    subdirs = [d for d in os.listdir(basedir) if os.path.isdir(basedir + d)]
    os.mkdir(targetdir)

# collect files
labeled_files = {}
for label in labels:
    labeled_files[label] = []
    for subdir in subdirs:
        curdir = basedir + subdir + "/" + label + "/"
        files = os.listdir(curdir)
        for file in files:
            if os.path.isfile(curdir + file) and file[0] != ".":
                labeled_files[label].append(
                    dict(path=curdir + file, file=file, subdir=subdir)
                )
totals = {k: len(d) for k, d in labeled_files.items()}
total = sum([d for k, d in totals.items()])
print("Found %d labeled files" % total)
for label in totals:
    print("%s: %d (%.2f%%)" % (label, totals[label], totals[label] / total * 100))

# coyp them over
for label, files in labeled_files.items():
    print("Copying %s..." % label)
    os.mkdir(targetdir + label + "/")
    for file in files:
        new_name = file["subdir"] + "_" + file["file"]
        shutil.copy(file["path"], targetdir + label + "/" + new_name)
print("done")
