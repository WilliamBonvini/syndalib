import os
import scipy.io
import numpy as np

from syndalib.linalg import normalize


def load_adelaide(path: str,
                  homogeneous: bool = False,
                  normalize_ds: bool = True):
    """

    :param path:
    :param homogeneous:
    :param normalize_ds:
    :return:
            ldata: each element is an np.ndarray, (npts, 4 or 6)
            llabels: each element is an np.ndarray, (npts, nm) - 0 corresponds to inlier; 1 corresponds to outlier
    """
    matfilenames = os.listdir(path)

    # all mat files
    ldata = []
    llabels = []

    for matfilename in matfilenames:
        matfile = scipy.io.loadmat(path + '/' + matfilename)

        # process data
        data = matfile['data']
        data = np.transpose(data)  # data now has shape (num correspondences, 6)
        if not homogeneous:
            data = data[:, [0, 1, 3, 4]]    # data now has shape (num correspondences, 4)

        if normalize_ds:
            data = np.transpose(normalize(data))
        ldata.append(data)

        # process labels
        labels = matfile['label']
        npts = labels.shape[1]
        labels = labels.reshape(npts, )
        nm = len(set(labels))
        pp_labels = np.ones((npts, nm))
        for corr in range(npts):
            label = labels[corr]
            pp_labels[corr][label] = 0
        pp_labels = np.delete(pp_labels, 0, axis=1)
        llabels.append(pp_labels)

    return ldata, llabels




