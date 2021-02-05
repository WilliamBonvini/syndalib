from typing import Type, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

#todo: get rid of it and of all this module
def plot_2d(points: Type[np.array]):
    """
    plots in 2D some points
    :param points: np.array, [(x_1,y_1,1), ..., (x_n,y_n,1)]
    :return:
    """
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.scatter(xs, ys, s=10)

"""
def plot_conic_curve(coefs: Union[List, np.ndarray],
                     x_range: Tuple[float, float] = (-10.0, 10.0),
                     y_range: Tuple[float, float] = (-10.0, 10.0),
                     resolution: Union[int, Tuple[int, int]] = 1000):
    
    plots a conic given its 6 coefficients
    :param coefs: list or np.ndarray. 6 conic parameters [x^2, xy, y^2, x, y, 1]
    :param x_range: interval of values of x for which the curve is to be plotted. default is (-10.0, 10.0)
    :param y_range: interval of values of y for which the curve is to be plotted. default is (-10.0, 10.0)
    :param resolution: how many points to be sampled in each of the 2 dimensions, default is 1000*1000
    :return:
    
    print("entrato in plot_conic_curve!!")
    print("aspetta un secondo, non mi dire che: coefs è un tensor D: ... type(coefs) = {}".format(type(coefs)))


    #print("a sto punto famo .numpy()!!")
    a, b, c, d, e, f = coefs
    print("unpacked coefs!! a = {}".format(a))

    if a == 0 and b == 0 and c == 0:
        raise Exception("Inputted linear equation")

    coefs_mat = np.array([[a, b / 2, d / 2],
                          [b / 2, c, e / 2],
                          [d / 2, e / 2, f]], dtype=float)

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
    for i in range(nx):
        for j in range(nx):
            point = np.array([xv[i, j], yv[i, j], 1], dtype=float)
            out = (point.dot(coefs_mat)).dot(point)
            if -1e-2 < out < 1e-2:
                x.append(point[0])
                y.append(point[1])

    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.scatter(x, y, s=5)
    plt.show()
    return
    


    points = []
    sym_points = []
    X = np.linspace(*x_range, num=800)

 

    if a*c < 0:
        print("hyperbola")

    # circle and ellipses case
    if b**2 - 4*a*c < 0:
        for x in X:
            delta = (b**2 * x**2) + e**2 + 2*b*e*x - 4*c*(a*x**2 + d*x + f)
            if delta < 0:
                continue
            if delta == 0:
                y = (-b*x-e)/2*c
                points.append((x, y))
            if delta > 0:
                y1 = (-b*x-e + np.sqrt(delta)) / 2*c
                y2 = (-b*x-e - np.sqrt(delta)) / 2*c
                points.append((x, y1))
                sym_points.append((x, y2))

                sym_points.reverse()
                sym_points.append(points[0])
                points.extend(sym_points)

    # parabola case
    elif a < 1e-5 or c < 1e-5:
        if a < 1e-5:
            Y = X
            for y in Y:
                num = c*y**2 + e*y + f
                denom = -b*y - d
                x = num / denom
                points.append((x, y))

        if c < 1e-5:
            for x in X:
                num = a*x**2 + d*x + f
                denom = -b*x-e
                y = num/denom
                points.append((x, y))

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    # plot curve
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.scatter(x, y, s=10)
    plt.show()
    """
