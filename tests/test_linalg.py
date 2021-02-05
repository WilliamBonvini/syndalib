from unittest import TestCase

from syndalib.linalg import compute_geometric_distance
from tests import *
import numpy as np
from syndalib import linalg
import tensorflow as tf


class Test(TestCase):
    def setUp(self) -> None:
        # first five samples are inliers; second five samples are outliers

        self.H = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])      # (3, 3)

        self.us = np.array([[1, 2, 1],
                            [3, 4, 1],
                            [5, 6, 1],
                            [7, 8, 1],
                            [9, 10, 1]])       # (5, 3)

        self.vs = np.transpose(np.matmul(self.H, np.transpose(self.us)))    # (5, 3)

        # put inliers in X:
        X = np.concatenate((self.us[:, 0:2], self.vs[:, 0:2]), axis=1)      # (5, 4)
        # put outliers in X:
        self.X = np.concatenate((X, np.random.rand(3, 4)), axis=0)          # (10, 4)

        # first 5 sample inliers, second 5 samples outliers
        self.w = np.concatenate((np.ones(5), np.zeros(5)))



    def test_conic_monomials(self):
        output = linalg.conic_monomials(circle_points_fixed)

        target = np.array([[1, 16, 4, 1, 4, 1],
                           [9, 1, 3, 3, 1, 1],
                           [1, 4, -2, 1, -2, 1],
                           [4, 1, -2, -2, 1, 1]], dtype=float)

        self.assertTrue((output == target).all())

    def test_circle_monomials(self):
        output = linalg.circle_monomials(circle_points_fixed)
        target = np.array([[17, 1, 4, 1],
                           [10, 3, 1, 1],
                           [5, 1, -2, 1],
                           [5, -2, 1, 1]])
        self.assertTrue((output == target).all())

    def test_circle_coefs_verbose(self):
        output_coefs = linalg.circle_coefs(*circle_params, verbose=True)
        print("output_coefs")
        print(output_coefs)

        target_coefs = np.array([1, 1, 0, -2, -2, -7], dtype=float)
        print("target_coefs")
        print(target_coefs)
        self.assertTrue((output_coefs == target_coefs).all())

    def test_circle_coefs_non_verbose(self):
        output_coefs = linalg.circle_coefs(*circle_params, verbose=False)
        print("output_coefs")
        print(output_coefs)

        target_coefs = np.array([1, -2, -2, -7], dtype=float)
        print("target_coefs")
        print(target_coefs)
        self.assertTrue((output_coefs == target_coefs).all())

    def test_dlt_coefs_both_args_nparray(self):
        coefs = linalg.dlt_coefs(circle_vandermonde_as_conic, circle_inliers_prob)
        target = circle_coefs_as_conic
        print("coefs = {}".format(coefs))
        print("target = {}".format(target))
        self.assertTrue(np.allclose(coefs, target))

    def test_dlt_coefs_both_args_tftensors(self):
        cvac = tf.constant(circle_vandermonde_as_conic)
        cip = tf.constant(circle_inliers_prob)
        coefs = linalg.dlt_coefs(cvac, cip)
        target = circle_coefs_as_conic
        print("coefs = {}".format(coefs))
        print("target = {}".format(target))
        self.assertTrue(np.allclose(coefs, target))

    def test_compute_geometric_distance_right_segm(self):
        result = compute_geometric_distance(self.X, self.w, self.H)
        print(result)
        assert result == 0

    def test_compute_geometric_distance_wrong_segm(self):
        wrong_weights = 1 - self.w
        result = compute_geometric_distance(self.X, wrong_weights, self.H)
        print(result)
        assert result > 0



