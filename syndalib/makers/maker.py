import abc
import math
import os
from abc import ABC
from random import uniform
from typing import Tuple, List
import syndalib.drawer as sydraw
import numpy as np
import pandas as pd
import scipy.io
from syndalib.utils.config import opts
from syndalib.utils.utils import (
    compute_num_inliers_per_model,
    plot_sample,
    convert_to_np_struct,
    convert_to_mat_struct,
)


class Maker(ABC):
    """
    abstract class. serves as reference to create all subclasses that
    will implement the generation of models of specific classes.
    """
    NUM_SAMPLES = None
    NUM_POINTS_PER_SAMPLE = None
    BASE_DIR = None
    TRAIN_DIR = None
    TEST_DIR = None
    CUR_DIR = None
    NUMBER_OF_MODELS = None
    OUTLIERS_PERC_RANGE = None
    NOISE_PERC_RANGE = None
    NOISE_PERC = None
    OUTLIERS_PERC = None
    INLIERS_RANGE = None
    IMG_BASE_DIR = None
    IMG_DIR = None
    MODEL = None
    PLOT = None
    SAVE_IMGS = None
    SAVE_NUMPY = None
    SAVE_MATLAB = None
    DT = np.dtype([("x1p", "O"), ("x2p", "O"), ("labels", "O")])

    @abc.abstractmethod
    def generate_random_model(self, n_inliers: int) -> List:
        """
        generate a random model. the generating process changes for each class.

        :param n_inliers: int, number of inliers to be generated
        :return: list, list of tuples (2d coordinates) of inliers
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def point(*args, **kwargs) -> Tuple:
        """
        generates a point of the random model of the specified class

        :param args: any parameter needed to generate model
        :param kwargs:
        :return: a tuple (x,y) representing a 2D point
        """
        pass

    def generate_random_sample(self, plot: bool = True):
        """
        generates a random sample of the desired class.

        :param: bool, true to plot sample, false otherwise. default is True.
        :return: np.ndarray (num inliers + num outliers, num coords + num models), shuffled sample

        in the previous version it returned:
            tuple, (x1,x2,labels) ---
        where:
        x1 is an np.ndarray (npps,);
        x2 is an np.ndarray (npps,);
        labels is an np.ndarray (npps, nm);
        """
        n_coords = 2
        tot_inliers = np.random.choice(Maker.INLIERS_RANGE)
        inliers_per_model = compute_num_inliers_per_model(tot_inliers, Maker.NUMBER_OF_MODELS)
        n_outliers = Maker.NUM_POINTS_PER_SAMPLE - tot_inliers
        sample = np.ones((tot_inliers + n_outliers, 2 + Maker.NUMBER_OF_MODELS))
        # GOTTA be (tot_inliers, n_coords)
        cursor = 0
        for i_model, n_inliers in zip(range(n_coords, n_coords + Maker.NUMBER_OF_MODELS), inliers_per_model):
            # model GOTTA be np.ndarray (n_inliers, n_coords)
            model_inliers = self.generate_random_model(n_inliers=n_inliers)
            sample[cursor:cursor + n_inliers, 0:n_coords] = model_inliers  # save coordinates
            sample[cursor:cursor + n_inliers, i_model] = np.zeros((n_inliers,))  # set inliers labels (inlier = 0)
            cursor = cursor + n_inliers

        # GOTTA be (n_outliers, n_coords)
        x_range = opts["outliers"]["x_r"]
        y_range = opts["outliers"]["y_r"]
        sample[cursor:, 0:n_coords] = sydraw.outliers_points(x_range=x_range, y_range=y_range, n=n_outliers, homogeneous=False)

        # todo: the plot_sample is also the one responsible to save the image. it should'nt be like this
        if plot:
            plot_sample(inliers=sample[0:cursor, 0:2],
                        outliers=sample[cursor:, 0:2],
                        inliers_per_model=inliers_per_model,
                        imgdir=Maker.IMG_DIR,
                        save_imgs=Maker.SAVE_IMGS)

        np.random.shuffle(sample)

        return sample

    def generate_dataset_fixed_nr_and_or(self) -> np.ndarray:
        """

        saves a .mat file in a structured fashioned folder with a fixed gaussian noise and a fixed outliers rate

        :return:  np.array, (ns, npps, n_coords + nm)
        """

        avg_num_inliers = math.floor(
            Maker.NUM_POINTS_PER_SAMPLE * (1.0 - Maker.OUTLIERS_PERC)
        )
        Maker.INLIERS_RANGE = list(range(avg_num_inliers - 2, avg_num_inliers + 2))

        # handle possibility that training or testing directory are empty
        if Maker.SAVE_IMGS:
            Maker.IMG_DIR = "{}/{}_no_{}_noise_{}" \
                            "".format(Maker.IMG_BASE_DIR, Maker.MODEL, int(Maker.OUTLIERS_PERC*100), Maker.NOISE_PERC)
            os.makedirs(Maker.IMG_DIR, exist_ok=True)

        samples = np.zeros((Maker.NUM_SAMPLES, Maker.NUM_POINTS_PER_SAMPLE, 2 + Maker.NUMBER_OF_MODELS))

        for i in range(Maker.NUM_SAMPLES):
            samples[i, ...] = self.generate_random_sample(plot=Maker.PLOT)

        data = samples[:, 0:2]
        labels = samples[:, 2:]

        if Maker.SAVE_NUMPY:
            # data_and_labels = {"data": data, "labels": labels, "outliers": Maker.OUTLIERS_PERC, "noise": Maker.NOISE_PERC}
            file_name = "{}/{}_no_{}_noise_{}".format(Maker.BASE_DIR, Maker.MODEL, int(Maker.OUTLIERS_PERC * 100), Maker.NOISE_PERC)
            np.save(file=file_name, arr=samples)

        if Maker.SAVE_MATLAB:
            dataset = np.array(samples, dtype=Maker.DT)
            dataset = np.array([dataset])
            print("shape dataset finale: {}".format(dataset.shape))
            print("shape dataset[0][5]: {}".format(dataset[0][5].shape))
            print("\n\n\n\ndataset [0][5]:\n{}".format(dataset[0][5]))
            print("shape dataset[0][5][0]: {}".format(dataset[0][5][0].shape))
            print("dataset[0][5][0]:\n{}".format(dataset[0][5][0]))
            # analizzo shape iniziale e shape finale. poi ristrutturo il codice di conseguenza
            # save it into a matlab file
            folder = "./{}/{}_no_{}_noise_{}.mat".format(Maker.BASE_DIR, Maker.MODEL, int(Maker.OUTLIERS_PERC * 100), Maker.NOISE_PERC)
            scipy.io.savemat(folder, mdict={"dataset": dataset, "outlierRate": Maker.OUTLIERS_PERC})

        return samples

    def generate_outliers(self, n_outliers: int) -> List:
        """
        generates a list of randomly sampled outliers

        :param n_outliers: int, number of outliers to be generated
        :return: list of tuples, list of randomly sampled outliers
        """
        x_min, x_max = opts["outliers"]["x_r"]
        y_min, y_max = opts["outliers"]["y_r"]

        outliers = [
            (uniform(x_min, x_max), uniform(y_min, y_max)) for _ in range(n_outliers)
        ]
        return outliers

    def start(self):
        for noise_stddev in Maker.NOISE_PERC_RANGE:
            for out_rate in Maker.OUTLIERS_PERC_RANGE:
                Maker.NOISE_PERC = noise_stddev
                Maker.OUTLIERS_PERC = out_rate
                dataset = self.generate_dataset_fixed_nr_and_or()
                return dataset
