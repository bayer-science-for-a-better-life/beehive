import numpy as np
from utils.vector import Vector, VectorCollection, consolidate_vectors
from utils.segmentation import Segment, Segmentation, consolidate_segments
from sklearn.cluster import DBSCAN


def get_neighbour_distances(D):
    '''Get shortest distances for each row in distance matrix excluding outliers
    D: eucl. distance matrix (n x n)
    '''
    min_Ds = np.nanmin(D[:-1], axis=1)

    # filter out distances which are probably not from direct neighbours
    IQR = np.quantile(min_Ds, 0.75) - np.quantile(min_Ds, 0.25)
    if IQR > 0:
        min_Ds = min_Ds[min_Ds < np.quantile(min_Ds, 0.5) + IQR]

    return min_Ds.mean(), min_Ds.std()


def get_neighbour_angles(A, D, max_distance):
    '''Get angles of neighbours from angle matrix
    Neighbours are identified by a corresponding distance matrix with
    a maximum distance
    D: eucl. distance matrix (n x n)
    A: corresponding angle matrix (same shape as D)
    max_distance: max distance for records to be considered neighbours
    '''
    angles = []
    for idx in range(D.shape[0] - 1):
        idxs_nbs = np.where(D[idx] <= max_distance)[0]
        if len(idxs_nbs) > 0:
            angles = angles + list(A[idx, idxs_nbs])
    return np.array(angles)


def predict_expected_angles(angles, expected_clusters=6):
    '''Predict expected angles from array of observed angles by DBSCAN
    angles: ndarray of observed angles
    expected_clusters: int of how many angle clusters to expect at max
    '''
    labels = DBSCAN(eps=np.pi / (2 * expected_clusters)) \
        .fit(angles.reshape(-1, 1)).labels_

    # get mean and std angles for each cluster, -1 are noisy clusters
    angle_means = []
    angle_stds = []
    for label in [d for d in set(labels) if d >= 0]:
        idxs = [i for i, d in enumerate(labels) if d == label]
        angle_means.append(angles[idxs].mean())
        angle_stds.append(angles[idxs].std())

    if len(angle_means) == 0:
        return [], 0

    # predict angles by scanning expected angle ranges
    max_std = max(angle_stds)
    arr = np.array(angle_means)

    def process_angle_range(tetha):
        found = arr[(arr >= tetha - 2 * max_std) & (arr <= tetha + 2 * max_std)]
        if found.shape[0] > 0:
            tetha = found[0]
        else:
            angle_means.append(tetha)

    # from one angle walk towards angle pi in expected step-size
    theta = angle_means[0]
    while theta <= np.pi:
        process_angle_range(theta)
        theta += np.pi / (expected_clusters / 2)

    # then walk towards angle -pi from one angle in expected step-size
    theta = angle_means[0]
    while theta >= -np.pi:
        process_angle_range(theta)
        theta -= np.pi / (expected_clusters / 2)

    # angles at pi can appear twice (pi and -pi), remove one
    angle_means = sorted(angle_means)
    if abs(angle_means[-1]) - abs(angle_means[0]) < np.pi / expected_clusters:
        angle_means = angle_means[1:]
    return angle_means, max_std


def predict_new_vectors(D, A, vectors, angles, angles_std, max_distance, mean_distance, kernel):
    '''From angles and distances predict new vectors
    For each vector look into expected directions and see if there is a direct
    neighbour. If there is no direct neighbour but further down this direction
    there is another vector, then we add a vector as direct neighbour.
    D: eucl. distance matrix (unecessary values should be Inf)
    A: angles matrix (of same shape as D)
    vectors: VectorCollection object of existing vectors
    angles: iter directions of expected neighbours
    angles_std: std of those directions
    max_distance: float of max eucl. distance in which we expect a neighbour
    mean_distance: float of mean eucl. distance in which we expect a neighbour
    kernel: float eucl. distance of kernel to use when searching neighbours
    '''

    def above_pi(t, h, l): return t <= h - 2 * np.pi or t >= l

    def below_pi(t, h, l): return t >= 2 * np.pi + l or t <= h

    def within_pi(t, h, l): return t <= h and t >= l

    out = VectorCollection()
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
                out.add(vectors[idx] + Vector.from_angle(angle, mean_distance))

    return out


def remove_border_segments(segments):
    '''Remove segments which happen reach over the border of the image.
    This can actually happen, if 2 small segments encapsulate a non-identified
    cell parallel to the border of the image. The predicted segment can be
    larger than the 2 encapsulating segments and thus span over the border
    of the image.
    '''
    out = Segmentation(img=segments.img)
    xmax = segments.img.shape[1]
    ymax = segments.img.shape[0]
    for i in range(len(segments)):
        s = segments[i]
        if min(s.xrange + s.yrange) > 0 and \
           max(s.xrange) <= xmax and \
           max(s.yrange) <= ymax:
            out.add(s)
    return out


def generate_segmentation(segments, n=2):
    '''Generate Segmentation by iteratively improving an initial segmentation
    Using an initial segmentation, we iteratively look at all the identified
    segments and see if they should have neighbours which are not yet identified.
    This is based on the usual neighbourhood of the existing segmentation.
    seg: Segmentation object of initial segmentation
    n: int number of iterations (0 ^= do nothing)
    '''
    for round in range(n):
        n_segments = len(segments)
        diams = np.ndarray((n_segments, 2))
        vectors = VectorCollection()
        for i in range(n_segments):
            s = segments[i]
            diams[i] = s.get_diams()
            vectors.add(Vector(*reversed(s.get_centroid())))

        A = vectors.get_angle_matrix()
        D = vectors.get_distance_matrix()
        D[D <= 0] = np.inf

        d_mean, d_std = get_neighbour_distances(D)
        neighbour_angles = get_neighbour_angles(A, D, d_mean + 2 * d_std)
        angles, angle_std = predict_expected_angles(neighbour_angles)
        vectors = predict_new_vectors(
            D=D, A=A, vectors=vectors, angles=angles,
            angles_std=2 * angle_std,
            max_distance=d_mean + d_std,
            mean_distance=d_mean,
            kernel=(d_mean + d_std) * 10)

        vectors = consolidate_vectors(vectors, d_mean / 2)
        pad = diams.mean() / 2
        for i in range(len(vectors)):
            v = vectors[i]
            segments.add(Segment([v.x - pad, v.x + pad], [v.y - pad, v.y + pad]))

        segments = consolidate_segments(segments, overlap=0.5)

    return remove_border_segments(segments)
