import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def batch_predictions(ds, model, batch_size, get_scores=False):
    """Run predictions of a dataset in batches

    ds: pytorch Dataset
    model: pytorch model with weights (and on device already)
    batch_size: int for batch size
    get_scores: bl whether to just return scores of first class
    """
    dl = DataLoader(ds, batch_size=batch_size)
    preds = []
    for X, Y in dl:
        scores = model(X)
        if get_scores:
            labels = scores[:, 0] - scores[:, 1]
        else:
            _, labels = torch.max(scores, 1)
        preds = preds + labels.tolist()
    return preds


def get_multi_confusion(labels, y_test, preds):
    """Generate Multiclass confusion matrix

    labels: sorted list of (unique) labels
    True label will be vertical, predicted label will be horizontal
    """
    multi_conf = np.zeros((len(labels), len(labels)))
    for i in range(len(y_test)):
        true_idx = labels.index(y_test[i])
        pred_idx = labels.index(preds[i])
        multi_conf[(true_idx, pred_idx)] += 1
    return multi_conf


def get_accuracy(conf, balanced=False):
    """Get accuracy from multi class confusion matrix

    balanced for balanced accuracy as average of recalls
    """
    if not balanced:
        return sum([conf[(i, i)] for i in range(conf.shape[0])]) / sum(sum(conf))
    return (
        sum([conf[i, i] / sum(conf[i]) for i in range(conf.shape[0])]) / conf.shape[0]
    )


def get_confusion_metrics(conf, label_names):
    """Calculate PPVs and TPRs from multiclass confusion matrix
    label_names should be in same order as confusion matrix
    """
    n = conf.shape[0]
    res = []
    for i in range(n):
        TP = conf[(i, i)]
        P = sum([conf[(j, i)] for j in range(n)])
        T = sum([conf[(i, j)] for j in range(n)])
        res.append(
            dict(
                label=label_names[i],
                PPV=TP / P if P > 0 else np.float("nan"),
                TPR=TP / T if T > 0 else np.float("nan"),
            )
        )
    return res


def get_roc(preds, truths):
    """Get ROC statistics

    preds: list of scores, high score ^= high probability of label '0'
    truths: list of labels '0' and '1'
    """
    df = pd.DataFrame(dict(score=preds, label=truths))
    df = df.sort_values("score", ascending=False)

    TPs = 0
    TNs = sum(df["label"] == 1)
    FPs = 0
    FNs = sum(df["label"] == 0)
    out = pd.DataFrame(
        dict(
            thresh=[0] * len(df),
            TP=[0] * len(df),
            TN=[0] * len(df),
            FP=[0] * len(df),
            FN=[0] * len(df),
        )
    )
    for i in range(len(df)):
        if df.iloc[i]["label"] == 0:
            TPs += 1
            FNs -= 1
        else:
            FPs += 1
            TNs -= 1
        out.iloc[i]["thresh"] = df.iloc[i]["score"]
        out.iloc[i]["TP"] = TPs
        out.iloc[i]["TN"] = TNs
        out.iloc[i]["FP"] = FPs
        out.iloc[i]["FN"] = FNs
    out["TPR"] = out["TP"] / (out["TP"] + out["FN"])
    out["FPR"] = out["FP"] / (out["FP"] + out["TN"])
    out["PPV"] = out["TP"] / (out["TP"] + out["FP"])
    out["FOR"] = out["FN"] / (out["FN"] + out["TN"])
    return out
