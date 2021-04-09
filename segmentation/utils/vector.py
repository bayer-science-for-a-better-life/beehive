import numpy as np
from sklearn.cluster import DBSCAN


def consolidate_vectors(vectors, eps):
    """Consolidate vectors which are close (eucl.) using DBSCAN
    vectors: VectorCollection with vectors to be consolidated
    eps: epsilon for DBSCAN (eucl.), how close to be the same grp
    """
    if len(vectors) < 1:
        return []
    coords = np.array([(v.y, v.x) for v in vectors])
    labels = DBSCAN(eps=eps, min_samples=1).fit(coords).labels_

    label2idxs = {label: [] for label in set(labels)}
    for i, d in enumerate(labels):
        label2idxs[d].append(i)

    consolidated_vectors = VectorCollection()
    for d in label2idxs:
        centroid = coords[label2idxs[d]].mean(axis=0)
        consolidated_vectors.add(Vector(centroid[1], centroid[0]))
    return consolidated_vectors


class Vector(object):
    """2D vector"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, o):
        return Vector(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Vector(self.x - o.x, self.y - o.y)

    def __repr__(self):
        return "<Vector (%.1f, %.1f) len %.1f theta %.1f />" % (
            self.x,
            self.y,
            self.len(),
            self.get_theta(),
        )

    def len(self):
        return abs(np.complex(self.x, self.y))

    def get_theta(self):
        return np.arctan2(self.y, self.x)

    @staticmethod
    def from_angle(tetha, len):
        x = np.cos(tetha)
        y = np.sin(tetha)
        a = len / Vector(x, y).len()
        return Vector(a * x, a * y)


class VectorCollection(object):
    """Combines a collection of vectors"""

    def __init__(self, vectors=None):
        if vectors is None:
            self.vectors = []
        else:
            self.vectors = vectors

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]

    def __repr__(self):
        return "<VectorCollection %d vectors />" % len(self.vectors)

    def add(self, vector):
        self.vectors.append(vector)

    def get_distance_matrix(self):
        """Calculate euclid. distance matrix for VectorCollection
        Form: array([[0, d, d],
                     [0, 0, d],
                     [0, 0, 0]])
        """
        x = [(d.x, d.y) for d in self.vectors]
        z = np.array([[complex(d[1], d[0]) for d in x]])
        return np.triu(abs(z.T - z))

    def get_angle_matrix(self):
        """Calculate radian angle matrix for VectorCollection
        Form: array([[0, t, t],
                     [t, 0, t],
                     [t, t, 0]])
        """
        n = len(self)
        X = np.ndarray((n, 1))
        Y = np.ndarray((n, 1))
        for i in range(n):
            v = self[i]
            X[i, 0] = v.x
            Y[i, 0] = v.y
        return np.arctan2(Y.T - Y, X.T - X)
