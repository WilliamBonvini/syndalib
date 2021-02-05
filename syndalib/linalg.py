from typing import Tuple
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


def conic_monomials(points):
    """
    given a set of points returns a matrix whose rows are the conic extensions for each point
    specifically, the terms are:
    x^2; xy; y^2; x; y; 1

    :param points: np.array, (num_points,). points are represented in homogeneous coordinates (x,y,z=1)
    :return: np.array, (num_points, 6)
    """
    n_points = points.shape[0]
    rows = np.zeros(shape=(n_points, 6))
    for i in range(len(points)):

        x = points[i][0]
        y = points[i][1]
        row = np.zeros(shape=(6,))
        row[0] = x**2
        row[1] = x*y
        row[2] = y**2
        row[3] = x
        row[4] = y
        row[5] = 1
        rows[i, :] = row

    return rows


def circle_monomials(points):
    """
    given a set of points returns a matrix whose rows are the conic extensions for each point
    specifically, the terms are:
    x^2 + y^2; x; y; 1

    :param points: np.array, (num_points,). points are represented in homogeneous coordinates (x,y,z=1)
    :return: np.array, (num_points, 4)
    """
    n_points = points.shape[0]
    rows = np.zeros(shape=(n_points, 4))
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        row = np.zeros(shape=(4,))
        row[0] = x ** 2 + y**2
        row[1] = x
        row[2] = y
        row[3] = 1
        rows[i, :] = row

    return rows


def dlt_coefs(vandermonde,
              weights=None,
              returned_type: str = "numpy"):
    """
    compute coefficients of a conic through direct linear mapping.
    vandermonde and weights arguments should be or both np.ndarrays or both tf.tensors


    :param vandermonde: (number of points, number of monomials),np.array or tf.tensor. each row contains monomials (e.g. for a conic: x^2 xy y^2 x y 1) of the corresponding point
    :param weights: (number of points,) np.array or tf.tensor. probability of belonging to the model for each row in the vandermonde matrix.
                    if all points belong to the model don't specify its value.
    :param returned_type: str, "numpy" returns an np.ndarray; "tensorflow" returns a tf.tensor
    :return: np.ndarray or tf.tensor (depending on the value of the parameter "type"), (number of monomials,), the coefficients computed via dlt
    """

    # tmp
    """
    if type(vandermonde) is not tf.constant:
        vandermonde = tf.constant(vandermonde)
    
    if type(weights) is not tf.constant and weights is not None:
        weights = tf.constant(weights)
        
    # this line is inserted only because in my project weights are np.ndarrays, and it conflicts with vandermonde (which is a tensor)
    weights = tf.constant(weights)
    """

    npps = weights.shape[0]

    if returned_type == "tensorflow":
        # preprocess weights

        weights = tf.cast(weights, dtype=tf.float64)
        weights = weights + K.random_uniform((npps,), 0, 1e-9, dtype=tf.float64)  # --> (npps, nm)
        weights = tf.linalg.normalize(weights, axis=0)[0]
        weights = tf.linalg.diag(weights)
        weighted_vander = tf.matmul(weights, vandermonde)

        U, S, V = tf.linalg.svd(weighted_vander)  # pay attention. tf.linals.svd returns directly V!! not V tranposed!

    elif returned_type == "numpy":
        weights = weights + np.random.normal(0, 1e-9)
        weights = weights / np.linalg.norm(weights)
        weights = np.diag(weights)
        weighted_vander = np.matmul(weights, vandermonde)
        U, S, VT = np.linalg.svd(weighted_vander)
        V = np.transpose(VT)

    else:
        raise Exception("Invalid argument for return_type")

    dlt_coefficients = V[:, -1]
    dlt_coefficients = dlt_coefficients * (1.0 / dlt_coefficients[0])  # want the x^2 and y^2 terms to be close to 1
    return dlt_coefficients


def circle_coefs(radius: float,
                 center: Tuple[float, float],
                 verbose: bool = True):
    """
    given a radius and a center it returns the parameters of the conic

    :param radius: radius of the circle
    :param center: center (x,y) of the circle
    :param verbose: True: returns 6 coefficients, corresponding to the terms (x^2; xy; x^2; x; y; 1)
                    False: returns 4 coefficients, corresponding to the terms (x^2+y^2; x; y; 1)
    :return: np.array, (num coefs,)
    """
    a = 1
    b = 0
    c = 1
    d = -2*center[0]
    e = -2*center[1]
    f = center[0]**2 + center[1]**2 - radius**2

    if verbose:
        return np.array([a, b, c, d, e, f], dtype=float)
    else:
        return np.array([a, d, e, f], dtype=float)


def veronese_map(points, n):
    """
    given a set of points and a degree n returns the veronese map of degree n for such points
    example:
    consider a veronese map of degree n for each homogeneous points (x,y,1) we'll have a row
    x^2 y^0 z^0 + x^1 y^1 z^0 + x^1 y^0 z^1 + x^0 y^2 z^0 + x^0 y^1 z^1 + x^0 y^0 z^2
    and, since z = 1, we can rewrite it as:
    x^2 + xy + x + y^2 + y + 1
    in tabular form:
      x    y    z
     i=2; j=0; k=0
     i=1; j=1; k=0
     i=1; j=0; k=1
     i=0; j=2; k=0
     i=0; j=1; k=1
     i=0; j=0; k=2


    :param points: np.array, (num_points,)
    :param n: degree of veronese map
    :return: np.array with veronese map, (num_points, veronese_columns)
    """
    cols = []
    i = n
    while i >= 0:
        j = n - i
        while j >= 0:
            k = n - i - j
            cols.append(points[:, 0]**i * points[:, 1]**j * points[:, 2]**k)
            j -= 1
        i -= 1
    return np.transpose(np.array(cols))


def homography_monomials(X: np.ndarray,
                         normalize: bool = True):
    """
    given correspondences (x,y) -> (u,v) it computes the monomials
    for the vandermonde matrix of the homography:
    xu; xv; x; yu; yv; y; u; v
    correspondences are to be given in non-hmoegeneous form --> X.shape is (4, num points)


    :param X: correspondences, np.ndarray, (4, num points)
    :param normalize: bool, normalize if true, default is true
    :return: Vandermonde matrix, np.ndarray, (num points, num terms)
    """

    N = X.shape[1]

    combinations = np.asarray(
        ((1, 0, 1, 0),  # xu
         (1, 0, 0, 1),  # xv
         (1, 0, 0, 0),  # x
         (0, 1, 1, 0),  # yu
         (0, 1, 0, 1),  # yv
         (0, 1, 0, 0),  # y
         (0, 0, 1, 0),  # u
         (0, 0, 0, 1)))  # v

    combinationsRow = combinations > 0

    nTerms = combinationsRow.shape[0] + 1
    V = np.zeros((N, nTerms))

    for ti in range(0, nTerms - 1):
        dimIdx = np.where(combinationsRow[ti, :])[0]
        tval = X[dimIdx, :]  # (dimIdx.shape[0], 512)
        degrees = combinations[ti, dimIdx]
        for r in range(1, tval.shape[0]):
            tval[r, :] = np.power(tval[r, :], degrees[r])

        # multiply together the variables x and u:
        tval = np.prod(tval, axis=0)
        V[:, ti] = tval

    V[:, nTerms - 1] = 1

    # normalize vandermonde matrix
    if normalize:
        V = np.divide(V, np.linalg.norm(V, axis=1).reshape((-1, 1)))

    return V


######### GENERIC

def weight_matrix(A: np.ndarray,
                  w: np.ndarray,
                  polarize: bool = False,
                  threshold: float = 0.5):
    """

    :param A: np.ndarray, (num points, num variables)
    :param w: np.ndarray, (num points,)
    :param polarize:
    :param threshold:
    :return:
    """

    if polarize:
        w = (w > threshold).astype(int)

    W = np.diag(w)
    WA = np.matmul(W, A)
    return WA









#### HOMOGRAPHY STUFF

def normalize(X):
    """
    normalize 2D-2D correspondences by applyng a mean subtraction and istropic scaling

    :param X: np.ndarray, ( npts, 4)
    :return: X_out, np.ndarray, (npts, 4)
    """
    npts = X.shape[0]

    x1 = X[:, 0]
    y1 = X[:, 1]
    x2 = X[:, 2]
    y2 = X[:, 3]

    # zero center
    tx1 = np.mean(x1)
    ty1 = np.mean(y1)
    tx2 = np.mean(x2)
    ty2 = np.mean(y2)

    # isotropic scaling
    xyz1 = np.transpose(X[:, 0:2])  # (2, npts)
    xyz1 = np.vstack((xyz1, np.ones(npts)))  # (3, npts)
    xyz2 = np.transpose(X[:, 2:])  # (2, npts)
    xyz2 = np.vstack((xyz2, np.ones(npts)))  # (3, npts)
    stddev1 = np.std(xyz1, 1)
    stddev2 = np.std(xyz2, 1)
    s1 = np.mean(stddev1) / np.sqrt(2)
    s2 = np.mean(stddev2) / np.sqrt(2)

    # Transformation matrices
    T1 = np.array([[1 / s1, 0, -tx1 / s1],
                   [0, 1 / s1, -ty1 / s1],
                   [0, 0, 1]])
    T2 = np.array([[1 / s2, 0, -tx2 / s2],
                   [0, 1 / s2, -ty2 / s2],
                   [0, 0, 1]])

    # transform
    xyz1 = np.dot(T1, xyz1)
    xyz2 = np.dot(T2, xyz2)
    X_out = np.concatenate((xyz1[0:2], xyz2[0:2]), axis=0)
    X_out = np.transpose(X_out)
    return X_out, T1, T2


def dlt_row_pair(x):
    """
    given a 2D-2D correspondence returns the two rows of the dlt matrix
    obtained by such correspondence
    :param x: np.ndarray, (4,)
    :return: rows, np.ndarray, (2, 9)
    """
    Ai = np.ones((2, 9))
    x1 = x[0]
    y1 = x[1]
    w1 = 1
    x2 = x[2]
    y2 = x[3]
    w2 = 1
    Ai[0] = np.array([0, 0, 0, -w2 * x1, -w2 * y1, -w2 * w1, y2 * x1, y2 * y1, y2 * w1])
    Ai[1] = np.array([w2 * x1, w2 * y1, w2 * w1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2 * w1])
    return Ai


def dlt_matrix(X):
    """

    :param X: np.ndarray, (npts, 4)
    :return: wmatrix of dlt, np.ndarray, (npts*2, 9)
    """
    npts = X.shape[0]

    A = np.ones((npts*2, 9))
    i = 0
    while i < npts:
        j = i * 2
        Ai = dlt_row_pair(X[i])
        A[j: j+2] = Ai
        i += 1

    return A


def compute_homography_from_dlt(A: np.ndarray,
                                compute_s: bool = False):
    """
    rename it into: retrieve_solution_from_svd
    apply svd to matrix A and returns solution (last column of V)

    :param A: np.ndarray, (npts*2, 9)
    :param compute_s: bool
    :return:
    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v = np.transpose(vh)
    tilde_h = v[:, -1]
    tilde_H = np.reshape(tilde_h, (3, 3))
    return tilde_H


def compute_H(X, w):
    """
    rename it into: dlt

   :param X: np.ndarray, (npts, 4)
   :param w: np.ndarray, (npts,)
   :return:
   """
    npts = X.shape[0]

    # normalize X
    X_norm, T1, T2 = normalize(X)

    # compute weighted dlt matrix
    A = dlt_matrix(X_norm)
    w = np.repeat(w, 2)
    A = weight_matrix(A, w, polarize=True)

    # retrieve homography from A
    H_norm = compute_homography_from_dlt(A=A)

    # denormalize H
    H = np.matmul(np.matmul(np.linalg.inv(T2), H_norm), T1)
    H = H / H[2, 2]
    return H


def compute_algebraic_distance(X, w, H):
    """


    :param X: np.ndarray, (npts, 4) --> computes avg;  || (4,) --> computes exact alg distance
    :param w: np.ndarray, (npts,)   --> computes avg;  || (1,) --> computes exact alg distance
    :param H: np.ndarray, (3,3)
    :return:
    """

    if len(X.shape) == 1:
        Ai = dlt_row_pair(X)
        err = np.dot(Ai, H.reshape(9))
        dist = np.sqrt(np.sum(err**2))
        return dist
    if len(X.shape) == 2:

        npts = X.shape[0]

        # algebraic error
        n_inliers = 0
        sum_dist = 0
        for i in range(npts):
            if w[i] == 1:
                n_inliers += 1
                Ai = dlt_row_pair(X[i])
                err = np.dot(Ai, H.reshape(9))
                dist = np.sqrt(np.sum(err**2))
                sum_dist += dist

        if n_inliers == 0:
            print("0 inliers for this sample")
            return 0

        alg_dist = sum_dist / n_inliers
        return alg_dist


def compute_geometric_distance(X: np.ndarray,
                               w: np.ndarray,
                               H: np.ndarray):
    """

    :param X: np.ndarray, (npts, 4)
    :param w: np.ndarray, (npts,)
    :param H: np.ndarray, (3,3)
    :return:
    """

    npts = X.shape[0]
    n_inliers = 0
    distance_sum = 0

    for i in range(npts):
        if w[i] == 1:
            n_inliers += 1
            u = X[i, 0:2]
            vpred = np.matmul(H, np.concatenate((u, np.ones(1))))    # (3,)
            vpred = vpred[0:2]
            Xpred = np.concatenate((u, vpred))
            diff = X[i] - Xpred
            distance = np.sum(diff**2)
            distance_sum += distance
    avg_dist = distance_sum / n_inliers
    return avg_dist
