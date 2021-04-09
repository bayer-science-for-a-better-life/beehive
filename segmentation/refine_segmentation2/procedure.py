import numpy as np
from utils.vector import Vector, VectorCollection, consolidate_vectors
from utils.segmentation import Segment, Segmentation, consolidate_segments
from sklearn.cluster import DBSCAN


def get_neighbour_distances(vectors, D):
    '''Get shortest distances excluding outliers from vectors
    vectors: VectorCollection object
    D: distance matrix of vectors
    '''
    min_Ds = D.min(axis=1)[:-1]

    # filter out distances which are probably not from direct neighbours
    IQR = np.quantile(min_Ds, 0.75) - np.quantile(min_Ds, 0.25)
    min_Ds = min_Ds[min_Ds < np.quantile(min_Ds, 0.5) + IQR]

    return min_Ds.mean(), min_Ds.std()


def get_neighbour_angles(vectors, D, max_distance):
    '''Calculate radian angles of neighbouring vectors
    vectors: VectorCollection object
    D: distance matrix of vectors
    max_distance: max distance for vectors to be considered neighbours
    '''
    angles = []
    for i1 in range(len(vectors) - 1):
        for i2 in range(i1 + 1, len(vectors)):
            if D[i1, i2] <= max_distance:
                angles.append((vectors[i2] - vectors[i1]).theta)
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


def predict_new_vectors(vectors, angles, angles_std, max_distance, mean_distance):
    '''From angles and distances predict new vectors
    For each vector look into expected directions and see if there is a direct
    neighbour. If there is no direct neighbour but further down this direction
    there is another vector, then we add a vector as direct neighbour.
    vectors: VectorCollection object of existing vectors
    angles: iter directions of expected neighbours
    angles_std: std of those directions
    max_distance: float of max eucl. distance in which we expect a neighbour
    mean_distance: float of mean eucl. distance in which we expect a neighbour
    '''

    def is_in_direction(v, angle):
        if angle + angles_std > np.pi:
            lo = angle + angles_std - 2 * np.pi
            return v.theta <= lo or v.theta >= angle - angles_std
        if angle - angles_std < -np.pi:
            hi = 2 * np.pi + angle - angles_std
            return v.theta >= hi or v.theta <= angle + angles_std
        return v.theta <= angle + angles_std and v.theta >= angle - angles_std

    def is_in_range(v):
        return v.len <= max_distance

    def is_closest_vector(v, closest):
        if closest is None:
            return True
        return v.len < closest.len

    # for each angle if there is a neighbouring vector in that direction
    # but it is not within range, estimate the missing vectors in between
    # and add them to the collection
    new_vectors = VectorCollection()
    for base_vector in vectors.copy():
        for angle in angles:
            has_nb = False
            in_range = False
            closest_vector = None

            for vector in vectors:
                diff = vector - base_vector
                if diff.len <= 0:
                    continue
                if is_in_direction(diff, angle):
                    has_nb = True
                    if is_in_range(diff):
                        in_range = True
                        break
                    elif is_closest_vector(diff, closest_vector):
                        closest_vector = diff

            if has_nb and not in_range:
                n_cells = int(round(closest_vector.len / mean_distance))
                ave_dist = closest_vector.len / n_cells
                for cell_i in range(n_cells):
                    new_vectors.add(base_vector + Vector.from_angle(angle, (cell_i + 1) * ave_dist))

    # t1 = time.time()
    # print('predicted %d new vectors from %d vectors, took %.2fs' % (len(new_vectors), len(vectors), t1 - t0))
    return new_vectors


def remove_border_segments(seg):
    '''Remove segments which happen reach over the border of the image.
    This can actually happen, if 2 small segments encapsulate a non-identified
    cell parallel to the border of the image. The predicted segment can be
    larger than the 2 encapsulating segments and thus span over the border
    of the image.
    '''
    seg_final = Segmentation(img=seg.img)
    for s in seg:
        if min(s.xrange + s.yrange) > 0 and \
           max(s.xrange) <= seg.img.shape[1] and \
           max(s.yrange) <= seg.img.shape[0]:
            seg_final.add(s)
    return seg_final


def generate_segmentation(segs):
    '''Generate Segmentation by iteratively improving an initial segmentation
    Using an initial segmentation, we iteratively look at all the identified
    segments and see if they should have neighbours which are not yet identified.
    This is based on the usual neighbourhood of the existing segmentation.
    seg: Segmentation object of initial segmentation
    '''
    vectors = VectorCollection()
    for s in segs:
        vectors.add(Vector(s.centroid[1], s.centroid[0]))
    D = vectors.get_distance_matrix()
    d_mean, d_std = get_neighbour_distances(vectors, D)
    angles = get_neighbour_angles(vectors, D, d_mean + 2 * d_std)
    predicted_angles, angle_std = predict_expected_angles(angles)
    pred_vectors = predict_new_vectors(
        vectors, predicted_angles,
        angles_std=2 * angle_std,
        max_distance=d_mean + d_std,
        mean_distance=d_mean)
    cons_vectors = consolidate_vectors(pred_vectors, d_mean / 2)
    mean_diam = np.mean([np.mean([s.xdiam, s.ydiam]) for s in segs])

    for v in cons_vectors:
        segs.add(Segment((v.y, v.x), *([mean_diam / 2] * 4)))
    segs = consolidate_segments(segs, overlap=0.5)

    return remove_border_segments(segs)
