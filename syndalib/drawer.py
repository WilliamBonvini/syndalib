import math
import random
import numpy as np
from typing import Tuple, List, Union
import os
import h5py
import tensorflow as tf


def outliers_points(x_range: Tuple[float, float],
                    y_range: Tuple[float, float],
                    n: int,
                    homogeneous: bool = True):
    """
    generates outliers in a user specified 2D square

    :param x_range: floats tuple, interval of x values
    :param y_range: floats tuple, interval if y values
    :param n: number of outliers
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :return: np array [(x_1,y_1,1),...,(x_np,y_np,1)]
    """

    outliers = []
    for _ in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        if homogeneous:
            outliers.append((x, y, 1))
        else:
            outliers.append((x, y))

    return np.array(outliers, dtype=float)


def circle_points(radius: float,
                  center: Tuple[float, float],
                  n: int,
                  noise: float = 0.0,
                  homogeneous: bool = True):
    """
    generates circle points given radius and center

    :param radius: radius
    :param center: center (x,y)
    :param n: number of points
    :param noise: gaussian noise standard deviation
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :return: np.ndarray, (n, n_coords): [(x_1,y_1,1),...,(x_np,y_np,1)]
    """
    points = []
    for _ in range(n):
        alpha = 2 * math.pi * random.random()
        x = center[0] + radius * math.cos(alpha) + np.random.normal(0, noise)
        y = center[1] + radius * math.sin(alpha) + np.random.normal(0, noise)

        if homogeneous:
            points.append((x, y, 1))
        else:
            points.append((x, y))

    return np.array(points)


def line_points(point_1: Tuple[float, float],
                point_2: Tuple[float, float],
                n: int,
                noise: float = 0.0,
                homogeneous: bool = True):
    """
    samples point from a line obtained by connecting two points
    :param point_1: (float, float)
    :param point_2: (float, float)
    :param n: number of points to be drawn
    :param noise: gaussian noise standard deviation, default is 0.0
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :return: np.array, shape = (n, n_coords)
    """
    x1, y1 = point_1
    x2, y2 = point_2
    m = (y1-y2)/(x1-x2)
    q = y1-m*x1
    points = []
    for _ in range(n):
        x = random.uniform(point_1[0], point_2[0])
        y = m*x + q + np.random.normal(0, noise)
        if homogeneous:
            points.append((x, y, 1))
        else:
            points.append((x, y))
    return np.array(points)


def corrupted_circle_points(radius: float,
                            center: Tuple[float, float],
                            n: int,
                            outliers_range: Tuple[Tuple[float, float], Tuple[float, float]],
                            outliers_perc: float,
                            noise: float = 0.0,
                            homogeneous: bool = True,
                            ):
    """
    array of points. the first n*(1.0 - outliers_perc) points belong to the specified circle, the remaining points
    are outliers, sampled from the specified square.

    :param radius: radius
    :param center: center (x,y)
    :param n: number of points
    :param outliers_range: tuple of tuples of floats, (x range, y range) of outliers. example: ((0.0,1.0), (0.0,1.0))
                           samples outliers from the unitary square in the cartesian plane
    :param outliers_perc: percentage of outliers with respect to the n points to be drawn
    :param noise: gaussian noise standard deviation
    :param homogeneous: bool, if true returns homogeneous coordinates, otherwise euclidean coordinates, default is true
    :return: np.array, shape = (n, n_coords)
    """
    n_in = int((1.0 - outliers_perc)*n)
    n_out = n - n_in
    circle = circle_points(radius=radius, center=center, n=n_in, noise=noise, homogeneous=homogeneous)
    outliers = outliers_points(*outliers_range, n=n_out, homogeneous=homogeneous)

    if homogeneous:
        n_coords = 3
    else:
        n_coords = 2

    points = np.zeros(shape=(n, n_coords))
    points[:n_in, :] = circle
    points[n_in:, :] = outliers
    return points


def conic_points(coefs: Union[List, np.ndarray],
                 x_range: Tuple[float, float] = (-10.0, 10.0),
                 y_range: Tuple[float, float] = (-10.0, 10.0),
                 resolution: Union[int, Tuple[int, int]] = 1000):
    """
    returns points of a conic given its 6 coefficients
    :param coefs: list of 6 conic parameters [x^2, xy, y^2, x, y, 1]
    :param x_range: interval of values of x for which the curve is to be plotted. default is (-10.0, 10.0)
    :param y_range: interval of values of y for which the curve is to be plotted. default is (-10.0, 10.0)
    :param resolution: how many points to be sampled in each of the 2 dimensions, default is 1000*1000
    :return: (x,y,), pair of coordinates of conic
    """
    if type(resolution) == int:
        nx = ny = resolution
    elif type(resolution) == Tuple:
        nx, ny = resolution
    else:
        raise Exception("resolution parameter should be a single int or a pair of ints")

    x = []
    y = []
    X = np.linspace(*x_range, nx)
    Y = np.linspace(*y_range, ny)
    xv, yv = np.meshgrid(X, Y)

    a, b, c, d, e, f = coefs
    if a == 0 and b == 0 and c == 0:
        raise Exception("Inputted linear equation")

    coefs_mat = np.array([[a, b / 2, d / 2],
                          [b / 2, c, e / 2],
                          [d / 2, e / 2, f]], dtype=float)

    for i in range(nx):
        for j in range(nx):
            point = np.array([xv[i, j], yv[i, j], 1], dtype=float)
            out = (point.dot(coefs_mat)).dot(point)
            if -1e-2 < out < 1e-2:
                x.append(point[0])
                y.append(point[1])
    return x, y





def homography_correspondences_from_modelnet():

    filename = "modelnet40_ply_hdf5_2048/ply_data_train0.h5"
    with h5py.File(filename, "r") as f:
        # List all groups
        keys = list(f.keys())
        data_key = keys[0]
        faceId_key = keys[1]
        label_key = keys[2]
        normal_key = keys[3]

        # Get the data
        data_original = list(f[data_key])
        labels = list(f[label_key])

    # save point clouds in structure
    data = np.zeros((3, 2048, 2048))
    for i in range(data.shape[2]):
        data[..., i] = np.transpose(data_original[i])

    # assign label to each sample
    f = open("modelnet40_ply_hdf5_2048/shape_names.txt", "r")
    label_names = f.readlines()
    label_names = [name.strip() for name in label_names]
    models_labels = []
    for i in range(len(labels)):
        models_labels.append(label_names[labels[i][0]])









