from scipy.spatial.transform import Rotation
import numpy as np
import pickle
import numpy.matlib


def simCamTransform(motion: str,
                    X: np.ndarray):
    """

    generation the rotation matrix and translation vector to apply a certain transformation (motion) to a point cloud

    :param motion: only "rotation" is implemented
    :param X: (6, npps)
    :return:
            rotation matrix, np.ndarray, (3,3)
            translation vector, np.ndarray, (3,1)
    """
    if motion == 'rotation':
        angles = 1.5 * (0.5 - np.random.rand(1, 3))  # it was 0.8, now it's 1.5
        R2 = Rotation.from_euler('zyx', angles, degrees=False)
        R2 = R2.as_matrix()[0]
        T2 = np.zeros((3, 1))

        return R2, T2


def simCamProj(camproj: str,
               X: np.ndarray,
               R: np.ndarray,
               t: np.ndarray):
    """
    generates a pair of 2D camera projections from a 3D point cloud.
    the first one is simply the projection on the z=1 plane of the point cloud.
    the second one is obtained by applying the rotation and translation given as inputs.

    :param camproj: kind of projection. only "uncalibrated" is implemented
    :param X: 3D point cloud, np.ndarray, (3, npps) (it is written in non-homogenenous 3D coordinates)
    :param R: rotation matrix, np.ndarray, (3,3)
    :param t: translation vector, np.ndarray, (3,1)
    :return:
        x1, 1st projection, np.ndarray, (2, npps)
        x2, 2nd projection, np.ndarray, (2, npps)
        K, intrinsic matrix np.ndarray (3,3)
    """

    if camproj == 'uncalibrated':
        f = 0.25 + 5 * np.random.rand(1, 1)

        P1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])

        P2 = np.concatenate((np.transpose(R), np.matmul(-np.transpose(R), t)), axis=1)

        N = X.shape[1]

        homo_X = np.concatenate((X, np.ones((1, N))), axis=0)

        lx1 = np.matmul(P1, homo_X)  # project points in 1st camera
        lx2 = np.matmul(P2, homo_X)  # project points in 2nd camera
        x1 = f * lx1[0:2, :] / numpy.matlib.repmat(lx1[2, :], 2, 1)
        x2 = f * lx2[0:2, :] / numpy.matlib.repmat(lx2[2, :], 2, 1)

        K = np.eye(3)
        K[0, 0] = f
        K[1, 1] = f  # k does only a 2D scaling ( no 2D translation and 2D shear)

        return x1, x2, K


def normalize(x: np.ndarray,
              alpha: float):
    """

    :param x: 2d point cloud, np.ndarray, (2, nps)
    :param alpha: float, don't know actually
    :return:
        x_out, normalized 2D point cloud, np.ndarray, (2,nps)
        TN, de-normalization matrix?
    """
    nps = x.shape[1]  # num points
    # zero-center
    t = np.mean(x, axis=1)
    x_out = x - np.repeat(t.reshape((-1, 1)), nps, axis=1)

    # don't know actually
    x_out[0, :] = x_out[0, :] / alpha

    # scaling in such a way that the RMS distance from the origin is sqrt(2) --> the "average point" is (1,1)
    d1 = np.mean(np.sqrt(np.sum(x_out ** 2, axis=0)))
    s = np.sqrt(2) / d1
    x_out = x_out * s

    # compute T (transformation matrix to de-normalize x_out)
    A = np.array([[s, 0, 0],
                  [0, s, 0],
                  [0, 0, 1]])

    B = np.array([[1/alpha, 0, -t[0] / alpha],
                  [      0, 1,         -t[1]],
                  [      0, 0,            1]])

    TN = np.matmul(A, B)
    return x_out, TN



def residual(x1, x2, Hn):
    """
    check the residual of the prediction for the homography Hn
    :param x1:
    :param x2:
    :param Hn:
    :return:
    """
    N = x1.shape[1]
    x2_pred = np.matmul(Hn, x1)
    error = 0
    for point in range(N):
        distance = np.sqrt(np.sum((x2[:, point] - x2_pred[:, point]) ** 2))
        error = error + distance
    error /= N
    return error


def save_obj(obj, name):
    """
    loads into obj the pkl file identified by "name"
    :param obj: obj into which to load file
    :param name: path to file
    :return:
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """

    :param name: path to pkl file
    :return: loaded object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

