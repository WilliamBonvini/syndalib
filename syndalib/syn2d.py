import os
from typing import Type, List, Union
import numpy as np
from syndalib.makers.circle_maker import CircleMaker
from syndalib.makers.conic_maker import ConicMaker
from syndalib.makers.ellipse_maker import EllipseMaker
from syndalib.makers.line_maker import LineMaker
from syndalib.makers.maker import Maker
from syndalib.utils.config import opts


def set_options(options: dict) -> None:
    """
    specify how to generate outliers and samples of specified classes.\n
    CLASS_TYPES = "outliers","circles", "lines", "ellipses", "conics".\n
    CLASS_OPTIONS are the values of the options dictionary and they vary with respect to the corresponding key.\n
    CLASS_OPTIONS always consists of a dictionary whose values are tuples of 2 floats
    (it represents the interval of values the specific parameter can take).\n
    the default options are the following:\n
    options = {
        "outliers": {
                    "x_r": (-2.5, 2.5),\n
                    "y_r": (-2.5, 2.5)
        },\n
        "circles": {
         \t   "radius_r": (0.5, 1.5),\n
         \t   "x_center_r": (-1.0, 1.0),\n
         \t   "y_center_r": (-1.0, 1.0),
        },\n
        "lines": {
                    "x_r": (-2.5, 2.5),\n
                    "y_r": (-2.5, 2.5)
        },\n
        "ellipses": {
            "radius_r": (0.5, 1.5),\n
            "x_center_r": (-1, 1),\n
            "y_center_r": (-1, 1),\n
            "width_r": (0.1, 1),\n
            "height_r": (0.1, 1)
        },\n
    }


    :param options: dict, a dictionary in the form {"CLASS_TYPE": CLASS_OPTIONS, ... }, where class_options is a class specific dictionary.
    :return:
    """

    for key in options.keys():
        opts[key] = options[key]


def generate_data(
    ns: int,
    npps: int,
    class_type: str,
    nm: int,
    outliers_range: List,
    noise_range: List,
    ds_name: str,
    is_train: bool = True,
    plot: bool = False,
    save_imgs: bool = False,
    save_numpy: bool = True,
    save_matlab: bool = False):
    """
    generates data corrupted by outliers and returns a numpy array.
    You can optionally save data in a .mat file in a structured fashion within the specified folder

    :param ns: int, total number of samples to be generated for each outlier rate
    :param npps: int, total number of points per sample
    :param class_type: str, models to be created. the possible models are: "circles","lines","ellipses","conics".
    :param nm: int, the number of models to be randomly sampled for each sample in the dataset, the default number is 1.
    :param outliers_range: list of floats, list of outliers rates (ex. outliers_range = [0.10,0.20] will generate samples with 10% and 20% outliers percentage
    :param noise_range: list of floats, list of stddevs of the gaussian noise to be added to the inliers (ex. noise range = [0.01, 0.02, 0.03] will generate samples with gaussian noise (0,0.01), (0,0.02) and (0, 0.03).
    :param ds_name: str, format of file "matlab" is the only one supported so far. default is "matlab".
    :param is_train: bool, true to write data in train folder, false to write data in test folder. default is True.
    :param plot: bool, if true shows the generated samples
    :param save_imgs: bool, if true saves some of the images of the generated samples
    :param save_numpy: bool, if true saves data and labels in a pandas dataframe within the specified folder
    :param save_matlab: bool, if true saves .mat file in structured fashion within a specified folder. default is False.
    :return: np.ndarray
    """

    Maker.NUM_SAMPLES = ns
    Maker.NUM_POINTS_PER_SAMPLE = npps
    Maker.OUTLIERS_PERC_RANGE = outliers_range
    Maker.MODEL = class_type
    Maker.NOISE_PERC_RANGE = noise_range
    Maker.DEST = format
    Maker.NUMBER_OF_MODELS = nm
    Maker.PLOT = plot
    Maker.SAVE_IMGS = save_imgs
    Maker.SAVE_NUMPY = save_numpy
    Maker.SAVE_MATLAB = save_matlab

    if Maker.SAVE_NUMPY or Maker.SAVE_MATLAB:
        # base dir
        train_or_test_dir = "train" if is_train else "test"
        Maker.BASE_DIR = "{}/nm_{}/{}/npps_{}/ns_{}/{}".format(class_type, nm, ds_name, npps, ns, train_or_test_dir)
        os.makedirs(Maker.BASE_DIR, exist_ok=True)

    if Maker.SAVE_IMGS:
        # create img dirs
        Maker.IMG_BASE_DIR = "{}/imgs".format(Maker.BASE_DIR)
        os.makedirs(Maker.IMG_BASE_DIR, exist_ok=True)

    maker = None
    if Maker.MODEL == "circles":
        maker = CircleMaker()
    elif Maker.MODEL == "lines":
        maker = LineMaker()
    elif Maker.MODEL == "ellipses":
        maker = EllipseMaker()
    elif Maker.MODEL == "conics":
        maker = ConicMaker()

    if maker is not None:
        dataset = maker.start()
    else:
        print('Incorrect name for model"')

    return dataset