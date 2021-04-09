import time
import numpy as np
from utils.vector import VectorCollection, Vector


def generate_vectors(n):
    X = np.random.uniform(low=0, high=100, size=n)
    Y = np.random.uniform(low=0, high=100, size=n)
    vc = VectorCollection()
    for i in range(n):
        vc.add(Vector(X[i], Y[i]))
    return vc


def predict_new_vectors(D, A, vectors, angles, angles_std, max_distance, mean_distance, kernel):

    def above_pi(t, h, l): return t <= h - 2 * np.pi or t >= l

    def below_pi(t, h, l): return t >= 2 * np.pi + l or t <= h

    def within_pi(t, h, l): return t <= h and t >= l

    new_vectors = VectorCollection()
    for idx in range(len(vectors)):
        d = D[idx]
        a = A[idx]

        for angle in angles:
            hi = angle + angles_std
            lo = angle - angles_std

            if hi > np.pi:
                is_in_direction = above_pi
            elif lo < -np.pi:
                is_in_direction = below_pi
            else:
                is_in_direction = within_pi

            idxs_ker = np.where(d <= kernel)[0]
            idxs_ang = [i for i in idxs_ker if is_in_direction(a[i], hi, lo)]

            if len(idxs_ang) > 0 and not any(d[idxs_ang] <= max_distance):
                    new_vectors.add(vectors[idx] + Vector.from_angle(angle, mean_distance))

    return new_vectors


def predict_new_vectors2(D, A, vectors, angles, angles_std, max_distance, mean_distance, kernel):

    def above_pi(t, h, l): return t <= h - 2 * np.pi or t >= l

    def below_pi(t, h, l): return t >= 2 * np.pi + l or t <= h

    def within_pi(t, h, l): return t <= h and t >= l

    def is_closest_vector(v, closest):
        if closest is None:
            return True
        return v.len() < closest.len()

    new_vectors = VectorCollection()
    for idx in range(len(vectors)):
        d = D[idx]
        a = A[idx]

        for angle in angles:
            hi = angle + angles_std
            lo = angle - angles_std

            if hi > np.pi:
                is_in_direction = above_pi
            elif lo < -np.pi:
                is_in_direction = below_pi
            else:
                is_in_direction = within_pi

            idxs_ker = np.where(d <= kernel)[0]
            idxs_ang = [i for i in idxs_ker if is_in_direction(a[i], hi, lo)]
            d_ang = d[idxs_ang]
            if len(idxs_ang) > 0 and not any(d_ang <= max_distance):
                idx_closest = idxs_ang[np.argmin(d_ang)]
                d_closest = d[idx_closest]
                n_cells = int(round(d_closest / mean_distance))
                ave_dist = d_closest / n_cells
                for cell_i in range(n_cells):
                    new_vectors.add(vectors[idx] + Vector.from_angle(angle, (cell_i + 1) * ave_dist))

    return new_vectors


angles = [np.pi, np.pi * 2 / 3, np.pi / 3, 0, -np.pi / 3, -np.pi * 2 / 3]


# > 99% of time is spend in innermost loop
# which is executed around 6M times for a 1000 vector collection

vc = generate_vectors(1000)
t0 = time.time()
D = vc.get_distance_matrix()
A = vc.get_angle_matrix()
D[D <= 0] = np.inf
res = predict_new_vectors(D, A, vc, angles, 0.1, 2, 1, 20)
print('%d -> %d: %.2fs' % (len(vc), len(res), time.time() - t0))
# 1000 -> 5065: 29.92s reduce scoping and duplicate calcs
# 1000 -> 5105: 25.80s optimize angle calculation
# 1000 -> 5086: 22.93s calc distances in outer loop w/ np
# 1000 -> 4148: 8.59s use kernel to only look at relevant distances
# 1000 -> 3789: 1.18s calc D first w/ np, optimized vectors
# 1000 -> 3806: 0.30s using angles matrix

vc = generate_vectors(1000)
t0 = time.time()
D = vc.get_distance_matrix()
A = vc.get_angle_matrix()
res = predict_new_vectors2(D, A, vc, angles, 0.1, 2, 1, 20)
print('%d -> %d: %.2fs' % (len(vc), len(res), time.time() - t0))
# 1000 -> 45354: 28.06s reduce scoping and duplicate calcs
# 1000 -> 39872: 1.46s applied changes from above

t0 = time.time()
x = 0
for i in range(len(vc) * len(vc) * len(angles)):
    x += 1
print('baseline %d iterations: %.2fs' % (x, time.time() - t0))
# baseline 6 000 000 iterations: 0.53s
