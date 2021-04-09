"""
# Evaluate trained models

see overall results, look at training progress
analyze predictions on validation set
"""
import torch
import json
import utils.plots as plots
import numpy as np
import pandas as pd
import utils.transforms as trans
import utils.evaluation as utils
from utils.models import smallCNN, simpleCNN, deepCNN
from utils.datasets import BeeDataset


# Validation Results
results = pd.read_csv("major_classes/results.csv")
plots.barplot(
    results,
    dict(x="Idx", y="Val-Rsq", hue="Start-LR"),
    "Val-Acc of Training Results using different Nets",
    "major_classes/plots/results_acc.png",
)
# Start-LR is main factor: 1e-3 is best
# model not so important apparently


# Best Results
q09 = round(results["Val-Rsq"].quantile(0.9), 3)  # 0.938
results[results["Val-Rsq"] >= q09]
# Idx	Model	Loss	Start-LR	Batch-Size	DR	Val-Rsq
# 4	4	simpleCNN	nlll	0.001	800	0.6	0.939089
# 5	5	deepCNN	mml	0.001	400	0.6	0.945003

row = results.iloc[5]


# Training Progress

# see progress file from best model
model_name = "{i}_{nn}_{loss}-loss_{lr:.0E}-lr_{bs}-bs_{dr}-dr".format(
    i=row["Idx"],
    nn=row["Model"],
    loss=row["Loss"],
    lr=row["Start-LR"],
    bs=row["Batch-Size"],
    dr=row["DR"],
)
progress = pd.read_csv("major_classes/training/%s.csv" % model_name)

plots.train_progress(
    progress,
    dict(x="Epoch", loss=["Train-Loss", "Val-Loss"], perf=["Train-Acc", "Val-Acc"]),
    title="Training Progress of best Cnn",
    save="major_classes/plots/progress_bestCnn.png",
)
# awesome training
# super small lead in val acc

# epoch of best model
progress["Val-Acc"].idxmax()  # 271


# Model
model = deepCNN(0, 8)
model.load_state_dict(
    torch.load("major_classes/models/%s.pkl" % model_name, map_location="cpu")
)
model.eval()
model.to("cpu")


# Load test data
ds_test = BeeDataset("dbs/master", "test", [trans.resize, trans.totensor])
truths, preds = utils.batch_predictions(ds_test, model, batch_size=200)


# adjust labels

# we excluded underrepresented classes 6, 9, 10
keep = [d not in [6, 9, 10] for d in truths]
truths = [d for d, k in zip(truths, keep) if k]
preds = [d for d, k in zip(preds, keep) if k]

# then we had different labels for the remaining classes
with open("major_classes/labels.json") as inf:
    labels = json.load(inf)

label_names = [d["name"] for d in labels]
label_map = {d["old_label"]: d["label"] for d in labels}
truths = [label_map[d] for d in truths]


# multi confusion
multi_conf = utils.get_multi_confusion(sorted(set(preds)), truths, preds)
acc = utils.get_accuracy(multi_conf)  # 0.939
multiconf(
    multi_conf,
    label_names,
    title="Confusion Matrix of Test Predictions",
    save="major_classes/plots/confusion_matrix.png",
)


# predictive values
metrics = utils.get_confusion_metrics(multi_conf, label_names)
plots.predictive_vals(
    metrics, title="Predictive Values", save="major_classes/plots/predictive_values.png"
)
pd.DataFrame(metrics)
# PPV	TPR	label
# 0	0.937500	0.940594	0 empty
# 1	0.955823	0.971429	1 egg
# 2	0.894253	0.858720	2 small
# 3	0.827089	0.864458	3 medium
# 4	0.966851	0.966851	4 big
# 5	0.991342	1.000000	5 capped
# 6	0.987915	0.987915	6 nectar
# 7	0.993399	0.980456	7 pollen
