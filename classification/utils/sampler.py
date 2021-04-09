"""Sampler Fractories

Sampler Factories for passing into utils.train.train_with_params()
They need to be callable with a list of labels
from which they are supposed to sample.
"""
from torch.utils.data.sampler import WeightedRandomSampler


class EqualDistribution(object):
    def __init__(self, replacement=True):
        self.replacement = replacement

    def __call__(self, labels):
        unique_labels = list(set(labels))
        n_per_label = [labels.count(d) for d in unique_labels]
        N = sum(n_per_label)
        w_per_label = [N / d for d in n_per_label]
        weights = []
        for label in labels:
            i = unique_labels.index(label)
            weights.append(w_per_label[i])
        return WeightedRandomSampler(
            weights, len(weights), replacement=self.replacement
        )
