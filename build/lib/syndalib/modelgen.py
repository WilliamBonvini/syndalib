import string
import os
from typing import Type, List

from syndalib.makers.circle_maker import CircleMaker
from syndalib.makers.conic_maker import ConicMaker
from syndalib.makers.ellipse_maker import EllipseMaker
from syndalib.makers.line_maker import LineMaker
from syndalib.makers.maker import Maker


def sampling_options(opts: dict):
    """
    example: opts = {"circles" : {"radius_range": (0.5, 1.5),
                                  "center_range": (1.0, 3.0)},
                     "line"    : {"slope_range": (1.0, 10.0),
                                  "intercept_range": (-10,10)}}

    :param opts: a dictionary in the form {"CLASS_TYPE": class_options, ... }
    where class_options is a class sepcific dictionary
    :return:


    """


def generate_data(
    ns: int,
    npps: int,
    class_type: str,
    nm: int,
    outliers_range: List,
    noise_range: List,
    ds_name: str,
    is_train: bool = True,
    format: str = "matlab",
):
    """
    generates .mat files in a structured fashion within the specified folder

    :param ns: int, total number of samples to be generated for each outlier rate
    :param npps: int, total number of points per sample
    :param class_type: str, models to be created. the possible models are: 'circles','lines','ellipses','conics'
    :param nm: int, the number of models to be randomly sampled for each sample in the dataset, the default number is 1
    :param outliers_range: list of floats, list of outliers rates
    :param noise_range: list of floats, list of stddevs of the gaussian noise to be added to the inliers
    :param ds_name: str, format of file "matlab", "numpy"
    :param is_train: bool, true to write data in train folder, false to write data in test folder. default is True
    :param format:
    :return:
    """

    Maker.NUM_SAMPLES = ns
    Maker.NUM_POINTS_PER_SAMPLE = npps
    Maker.OUTLIERS_PERC_RANGE = outliers_range
    Maker.MODEL = class_type
    Maker.NOISE_PERC_RANGE = noise_range
    Maker.DEST = format


    Maker.NUMBER_OF_MODELS = nm

    # create basedir
    basedir = class_type
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    basedir += "/nm_" + str(nm)
    if not (os.path.exists(basedir)):
        os.mkdir(basedir)
    basedir += "/" + ds_name
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    basedir += "/npps_" + str(npps)
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    basedir += "/ns_" + str(ns)
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    Maker.BASE_DIR = basedir

    # create img dirs
    imgbasedir = Maker.BASE_DIR
    if Maker.TRAIN_DIR != "":
        imgbasedir += "/" + "train"
    if Maker.TEST_DIR != "":
        imgbasedir += "/" + "test"
    if not os.path.exists(imgbasedir):
        os.mkdir(imgbasedir)
    imgbasedir += "/" + "imgs"
    if not os.path.exists(imgbasedir):
        os.mkdir(imgbasedir)

    Maker.IMG_BASE_DIR = imgbasedir

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
        maker.start()
    else:
        print('Incorrect name for "model"')
