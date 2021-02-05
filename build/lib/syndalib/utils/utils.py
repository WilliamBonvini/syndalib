import string
from math import floor
from typing import List, Tuple

import matplotlib.pyplot as plt
import random as rand
import os
import numpy as np


def getRanges(nSamplesPerOR: int):
    """

    :param nSamplesPerOR:
    :return:
    """
    xr = 1
    yr = 1
    rr = 1
    turn = 0
    while xr * yr * rr < nSamplesPerOR:
        if turn == 0:
            xr *= 2
        elif turn == 1:
            yr *= 2
        elif turn == 2:
            rr *= 2
            turn = -1
        turn += 1

    return [xr, yr, rr]


def checkExistenceAndCreate(path: string):
    """

    :param path: takes full path of mat file, included the name of the file
    :return:
    """
    folders = path.split("/")
    folders = folders[1:-1]  # dont' care about the '.' nor the name of the file
    path = folders[0]
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + "/" + folders[1]
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + "/" + folders[2]
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + "/" + folders[3]
    if not os.path.exists(path):
        os.mkdir(path)


def convert_to_np_struct(inliers: List, outliers: List) -> np.ndarray:
    """
    merges list of inliers and list of outliers into an np.ndarray and returns it

    :param inliers: list, [ [(x11,y11), ..., (x1n, y1n)], ..., [(xm1,ym1), ..., (xmn,ymn)] ] one list for each model
    :param outliers: list, [(x1,y1), ..., (xn,yn)].
    :return: np.ndarray (num inliers + num outliers, num coords + num models),  NOT shuffled sample
    """

    # convert inliers to np array
    inliers = np.array([np.array(model_inliers) for model_inliers in inliers])  # np.ndarray (num models, num points)

    # obtain number of models
    num_models = inliers.shape[0]

    # count all inliers by summing inliers in each model
    num_inliers = 0
    for i_model in range(num_models):
        num_inliers += inliers[i_model].shape[0]

    # obtain number of coordinates (in the circle case it is: 2)
    n_coordinates = inliers[0].shape[1]

    # create an array of ones that I'll modify step by step
    # it is an array of ones because I'll have that some point will be inlier for a model but outliers (ones) for others
    inliers_out = np.ones((num_inliers, n_coordinates + num_models))
    for i_model in range(num_models):
        # find start index row for this model
        start_i = 0
        for i in range(i_model):
            start_i += inliers[i].shape[0]
        num_inliers_in_model = inliers[i_model].shape[0]
        # find end index row for this model
        end_i = start_i + num_inliers_in_model
        inliers_out[start_i:end_i, 0:n_coordinates] = inliers[i_model]
        inliers_out[start_i:end_i, n_coordinates + i_model] = np.zeros(
            shape=(num_inliers_in_model,)
        )

    # set value to 1 for outliers
    outliers_out = np.ones((len(outliers), n_coordinates + num_models))
    outliers_out[:, 0:n_coordinates] = outliers

    # concatenate arrays and shuffle
    sample = np.concatenate((inliers_out, outliers_out), axis=0)

    return sample


def convert_to_mat_struct(sample: np.ndarray) -> Tuple:
    """
    converts np.ndarray into triplet (x1p, x2p, labels)

    :param sample: np.ndarray, sample coded as 2d points + their label
    :return: a triplet made of correspondences + labels, the kind of struct matlab deals with
    """
    print("convert_to_mat_struct_input = {}".format(sample.shape))
    sample = sample.T
    x1p = np.array([elem for elem in sample[0, :]], dtype=np.float32)
    x2p = np.array([elem for elem in sample[1, :]], dtype=np.float32)
    num_models = sample.shape[0] - 2

    num_points_per_sample = sample.shape[1]
    labels = np.zeros(shape=(num_points_per_sample, num_models))
    for i_model in range(num_models):
        labels[:, i_model] = np.array([elem for elem in sample[2 + i_model, :]], dtype=np)
    # se dovessi avere problemi con il single model fitting forse mi basta deglutire la seconda dimensione di labels
    # oltre al commento sopra dovrei evitare di fare la seguente transpose immagino
    labels = labels.T
    circle_mat = (x1p, x2p, labels)
    print("convert_to_mat_struct output shapes = \n[0] is {}\n[1] is {}\n[2] is {}".format(circle_mat[0].shape, circle_mat[1].shape, circle_mat[2].shape))
    return circle_mat


def compute_num_inliers_per_model(tot_num_inliers, num_of_models):
    """

    :param tot_num_inliers: total number of inliers in the whole sample
    :param num_of_models: the number of models to be drawn in each sample
    :return: a list that contains the number of inlier points for each model
    """
    n_inliers = []
    remaining = tot_num_inliers
    remaining_n_of_models = num_of_models
    while remaining_n_of_models > 0:
        cur_model_n_nliers = floor(remaining / remaining_n_of_models)
        n_inliers.append(cur_model_n_nliers)
        remaining -= cur_model_n_nliers
        remaining_n_of_models -= 1
    return n_inliers


def plot_sample(inliers:np.ndarray,
                outliers: np.ndarray,
                inliers_per_model: List,
                imgdir: str,
                save_imgs: bool):
    """
    saves and plots samples
    :param inliers: np.ndarray, (tot_num inliers, n_coords)
    :param outliers: np.ndarray, (num_outliers, n_coords)
    :param inliers_per_model: list of ints, inliers is an ordered array. the first inliers_per_model[0] points belong
                              to the first model, the second inliers_per_moel[1] points belong to the second model,
                              and so on
    :param imgdir: str, path of the directory in which to save images
    :param save_imgs: if true saves image once in a while
    :return: nothing
    """
    cursor = 0
    for n_inliers in inliers_per_model:
        x = inliers[cursor:cursor + n_inliers, 0]
        y = inliers[cursor:cursor + n_inliers, 1]
        cursor += n_inliers
        plt.scatter(x, y, s=2)

    x = outliers[:, 0]
    y = outliers[:, 1]
    plt.scatter(x, y, s=2)
    plt.grid(color="k", linestyle=":", linewidth=1)
    plt.axes().set_aspect("equal", "datalim")
    plt.gca().set_aspect("equal", adjustable="box")

    # save plot once in a while
    random_number = rand.randrange(10000)
    if save_imgs and random_number % 20 == 0:
        folder = imgdir + "/" + str(rand.randrange(10000)) + ".png"
        plt.savefig(folder)

    # show it
    plt.show()
