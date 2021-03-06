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

    if returned_type == "numpy":

        print("cani")
        print("{}\n{}".format(type(weights), type(vandermonde)))
        # preprocess weights

        weights = tf.cast(weights, dtype=tf.float64)
        weights = weights + K.random_uniform((npps,), 0, 1e-9, dtype=tf.float64)  # --> (npps, nm)
        weights = tf.linalg.normalize(weights, axis=0)[0]
        weights = tf.linalg.diag(weights)
        weighted_vander = tf.matmul(weights, vandermonde)

        U, S, V = tf.linalg.svd(weighted_vander)  # pay attention. tf.linals.svd returns directly V!! not V tranposed!

    elif returned_type == "tensorflow":
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


