import math
from random import uniform
from typing import List

import numpy as np
import random as rand
from syndalib.makers.maker import Maker
from syndalib.utils.config import opts
import syndalib.drawer as sydraw


class CircleMaker(Maker):
    """
    Maker implementation for circles sampled from a 2D space.
    you can define the 2D space by calling the function syndalib.syn2d(...).
    """
    def __init__(self):
        Maker.MODEL = "circles"
        super(CircleMaker, self).__init__()

    def generate_random_model(self, n_inliers: int = None) -> List:
        """
        generates points belonging to a randomly sampled model

        :param n_inliers: number of inlier points for this specific model
        :return: np.ndarray, (n_inliers, n_coords)
        """
        x_c_min, x_c_max = opts["circles"]["x_center_r"]
        y_c_min, y_c_max = opts["circles"]["y_center_r"]
        r_min, r_max = opts["circles"]["radius_r"]

        x_center = uniform(x_c_min, x_c_max)
        y_center = uniform(y_c_min, y_c_max)
        radius = uniform(r_min, r_max)

        xy = sydraw.circle_points(radius=radius, center=(x_center, y_center), n=n_inliers, noise=CircleMaker.NOISE_PERC, homogeneous=False)

        return xy

    @staticmethod
    def point(h: float, k: float, r: float, noise: float):
        """
        draws a random point belonging to a circle with center (h,k) and radius r

        :param h: x center of circle
        :param k: y center of circle
        :param r: radius of circle
        :param noise: stddev of gaussian noise
        :return:
        """

        theta = rand.random() * 2 * math.pi
        x = h + math.cos(theta) * r
        noise_x = np.random.normal(0, noise)
        x = x + noise_x
        y = k + math.sin(theta) * r
        noise_y = np.random.normal(0, noise)
        y = y + noise_y
        return x, y
