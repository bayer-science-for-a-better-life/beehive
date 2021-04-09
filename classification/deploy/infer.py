import torch
from models import MajorClasses


def get_batches(n_total, batch_size=100):
    '''Get ranges for dividing a set into batches of maximum size
    Returns a list of tuples with (start, stop) indexes needed to divide a
    list of length n_total into parts of maximum size batch_size (=100 default)
    '''
    out = []
    running_n = n_total
    upper = 0
    lower = 0
    while running_n > 0:
        lower = upper
        upper = lower + batch_size if running_n > batch_size else lower + running_n
        out.append((lower, upper))
        running_n += lower - upper
    return out


def infer_bees(dataset):
    major_classes = MajorClasses()

    out = []
    for part in get_batches(len(dataset)):
        ds_part = dataset[part[0]:part[1]]
        X = torch.stack([d for d in ds_part])
        classes = major_classes(X)
        for i in range(len(ds_part)):
            out.append(dict(class_caller=classes[i]))
    return out
