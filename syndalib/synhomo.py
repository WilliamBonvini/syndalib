import os
from typing import List, Tuple, Union

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from syndalib.projections.cameraops import simCamTransform, simCamProj, normalize, save_obj
from syndalib.utils.enums import COLORS, OUTLIERS_COLOR
from syndalib.utils.utils import compute_num_inliers_per_model


def load_modelnet_data_and_labels(path_to_modelnet: str,
                                  path_to_shape_names: str) -> Union[np.ndarray, List[str]]:
    """
    loads data and a list of names, one for each point cloud in modelnet40

    :param path_to_modelnet: smng like
    :param path_to_shape_names: smng like "modelnet40_ply_hdf5_2048/shape_names.txt"
    :return: data: np.ndarray, (3, 2048, 2048)
             models_labels: list of str, one name for each of the 2048 point clouds in modelnet40
    """

    with h5py.File(path_to_modelnet, "r") as f:
        keys = list(f.keys())
        data_key = keys[0]
        label_key = keys[2]
        data_original = list(f[data_key])
        labels = list(f[label_key])

    # process data
    data = np.zeros((3, 2048, 2048))
    for i in range(data.shape[2]):
        data[..., i] = np.transpose(data_original[i])

    # assign label to each sample
    f = open(path_to_shape_names, "r")
    label_names = f.readlines()
    label_names = [name.strip() for name in label_names]

    models_labels = []
    for i in range(len(labels)):
        models_labels.append(label_names[labels[i][0]])

    return data, models_labels


def generate_homographies(data: np.ndarray,
                          samples_names: List[str],
                          outliers_range: List[float],
                          ns: int,
                          npps: int,
                          nm: int,
                          save_imgs: bool = False,
                          save_datasets: bool = False) -> None:
    """
    generates num_homos homographies by 2D-projecting in 2 different views a point cloud X chosen from data

    :param data:  array of 3D point cloud, np.ndarray, (3, nps, num point clouds)
    :param samples_names: list of str, a name for each point cloud in data
    :param outliers_range: list of floats, example: [0.10, 0.20]
    :param ns: number of samples to be generated
    :param npps: number of points per sample (all homographies and outliers included)
    :param nm: number of homographies per sample to be generated (number of models)
    :param save_imgs: bool, if True saves imgs of correspondences
    :param save_datasets: bool, if True saves datasets in pkl files
    :return: list of homographies
    """

    tot_point_clouds = data.shape[-1]
    tot_points_per_point_cloud = data.shape[1]

    list_of_datasets = []
    for outliers_rate in outliers_range:
        dir = "homographies/{}".format(nm)
        img_dir = dir + "/imgs"
        os.makedirs(img_dir, exist_ok=True)

        dataset = {}
        dataset["outliers rate"] = outliers_rate
        dataset["data"] = {}
        for i_sample in range(ns):
            pcs_selected = np.random.permutation(tot_point_clouds)[:nm]

            x1s = np.zeros((3, npps))
            x2s = np.zeros((3, npps))
            labels = np.ones((nm, npps))

            num_points_per_model = compute_num_inliers_per_model(npps, nm)

            outliers_points1 = np.array([]).reshape((3, 0))
            outliers_points2 = np.array([]).reshape((3, 0))
            if save_imgs:
                fig, (view1, view2) = plt.subplots(1, 2)
                fig.suptitle('2 views of objects')

            curr_idx = 0

            for i, i_pc in zip(np.arange(nm), pcs_selected):

                npm = num_points_per_model[i]  # num points of model
                corr_indexes = np.random.permutation(tot_points_per_point_cloud)[:npm]
                X = data[:, corr_indexes, i_pc]
                x1ph, x2ph, TN1, TN2, F, Fn, E, K, R, t, H, Hn = compute_2d_views_and_homography(X)

                # Generate outliers by mixing
                n_outliers = int(npm * outliers_rate)
                matches = np.repeat(np.array([np.arange(0, npm, 1)]), 2, axis=0)
                perm = np.random.permutation(npm)
                outliers_indexes = perm[:n_outliers]
                inliers_indexes  = perm[n_outliers:]
                matches[1, outliers_indexes] = np.random.permutation(npm)[:n_outliers]
                outliers_points1 = np.hstack((outliers_points1, x1ph[:, outliers_indexes]))
                outliers_points2 = np.hstack((outliers_points2, x2ph[:, outliers_indexes]))

                # arrange points according to matches
                x1ph = x1ph[:, matches[0, :]]
                x2ph = x2ph[:, matches[1, :]]

                # save labels
                labels_m = np.zeros(npm)
                labels_m[outliers_indexes] = 1
                labels[i, curr_idx: curr_idx + npm] = labels_m

                if save_imgs:
                    view1.scatter(x1ph[0][inliers_indexes], x1ph[1][inliers_indexes], s=5, c=COLORS[i], label=samples_names[i_pc])
                    view2.scatter(x2ph[0][inliers_indexes], x2ph[1][inliers_indexes], s=5, c=COLORS[i], label=samples_names[i_pc])

                # save correspondences
                x1s[:, curr_idx: curr_idx + npm] = x1ph
                x2s[:, curr_idx: curr_idx + npm] = x2ph

                curr_idx += npm

            # put data in dataset
            dataset["data"][i_sample] = {}
            dataset["data"][i_sample]["x1s"] = x1s
            dataset["data"][i_sample]["x2s"] = x2s
            dataset["data"][i_sample]["labels"] = labels
            for m in range(nm):
                key = "info model " + str(m+1)
                dataset["data"][i_sample][key] = [TN1, TN2, F, Fn, E, K]

            if save_imgs and i_sample % 20 == 0:
                view1.scatter(outliers_points1[0], outliers_points1[1], s=5, c=OUTLIERS_COLOR, label="outliers")
                view2.scatter(outliers_points2[0], outliers_points2[1], s=5, c=OUTLIERS_COLOR, label="outliers")
                view1.legend()
                view2.legend()
                plt.savefig(img_dir + "/" + str(i_sample) + ".png")
                plt.close()

        # save dataset
        if save_datasets:
            save_obj(dataset, dir + "/" + str(outliers_rate) + "_2d_homs")
        list_of_datasets.append(dataset)

    return list_of_datasets


def compute_2d_views_and_homography(X: np.ndarray):
    """
    starting from a 3D point clouds computes a pair of 2D projections explainable by a homography

    apply random rotation R1 to X,
    then generates 2 2D-projections of X:
    - the first  one is a projection of X rotated by Rglobal,
    - the second one is a projection obtained by further applying R and t

    :param X: 3D point cloud, np.ndarray, (3, npts)
    :return:
        x1ph: 1st projection in homogeneous coordinates, np.ndarray, (3, nps)
        x2ph: 2nd projection in homogeneous coordinates, np.ndarray, (3, nps)
        TN1:  de-normalization matrix for 1st projection supposedly, np.ndarray, (3,3)
        TN2:  de-normalization matrix for 2nd projection supposedly, np.ndarray, (3,3)
        Forg: identity, np.ndarray, (3,3)
        Fn:   identity, np.ndarray, (3,3)
        E:    identity, np.ndarray, (3,3)
        K,    intrinsic matrix, np.ndarray, (3,3)
        R,    motion matrix from view 1 to view 2, np.ndarray, (3,3)
        t,    translation vector from view 1 to view 2, np.ndarray, (3,3)
        H,    homography matrix, np.ndarray, (3,3)
        Hn,   homography matrix to be applied to normalized data, np.ndarray, (3,3)
    """

    motion = 'rotation'
    camproj = 'uncalibrated'
    npts = X.shape[-1]

    # randomly rotate X
    angles = 0.8 * (0.5 - np.random.rand(1, 3))
    Rglobal = Rotation.from_euler('zyx', angles, degrees=False)
    Rglobal = Rglobal.as_matrix()[0]
    X = np.matmul(Rglobal, X)

    # put in front of camera 1
    X = X + np.repeat(np.transpose(np.array([[0, 0, 2]])), npts, axis=1)
    # mix homographies
    # skipped

    # generate motion
    R, t = simCamTransform(motion, X)

    # generate camera projection
    x1p_org, x2p_org, K = simCamProj(camproj, X, R, t)

    x1p, TN1 = normalize(x1p_org, 1)
    x2p, TN2 = normalize(x2p_org, 1)

    x1ph = np.ones((3, npts))
    x1ph[0:2] = x1p
    x2ph = np.ones((3, npts))
    x2ph[0:2] = x2p

    # add noise
    noise = 6 / 512
    x1ph[0:2, :] = x1ph[0:2, :] + (2 * np.random.rand(2, npts) - 1) * noise
    x2ph[0:2, :] = x2ph[0:2, :] + (2 * np.random.rand(2, npts) - 1) * noise

    # compute homography from rotation
    H = np.matmul(K, np.matmul(R, np.linalg.inv(K)))
    Hn = np.matmul(TN2, np.matmul(np.linalg.inv(H), np.linalg.inv(TN1)))

    E = np.eye(3)
    Forg = np.eye(3)
    Fn = np.eye(3)

    return x1ph, x2ph, TN1, TN2, Forg, Fn, E, K, R, t, H, Hn
