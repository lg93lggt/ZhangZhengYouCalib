import numpy as np
from scipy.optimize import leastsq

class Homography(object):
    """
    A sovler to caulate Homography martix for one image.
    """

    def __init__(self, world_points, image_points):
        """
        Argin:
            world_points: A (points_num, 2) array.
            image_points: A (points_num, 2) array which corresponds to world_points.
        """
        points_num = world_points.shape[0]
        M = np.zeros((2 * points_num, 8))
        b = np.zeros((2 * points_num, 1))

        for i in range(points_num):
            index1 = 2 * i 
            index2 = index1 + 1

            M[index1, 0] = world_points[i, 0] # X
            M[index1, 1] = world_points[i, 1] # Y
            M[index1, 2] = 1
            M[index1, 6] = -(image_points[i, 0] * world_points[i, 0]) # -uX
            M[index1, 7] = -(image_points[i, 0] * world_points[i, 1]) # -uY

            M[index2, 3] = world_points[i, 0] # X
            M[index2, 4] = world_points[i, 1] # Y
            M[index2, 5] = 1
            M[index2, 6] = -(image_points[i, 1] * world_points[i, 0]) # -vX
            M[index2, 7] = -(image_points[i, 1] * world_points[i, 1]) # -vY


            b[index1, :] = image_points[i, 0] # u
            b[index2, :] = image_points[i, 1] # v

        self.M = np.matrix(M)
        self.b = np.matrix(b)
        del(M, b)
        return

    def solve_H_OLS(self):
        """
        Solve H by Ordinary Least Square.
            M * h = b
            argmin(b - M * h)
            h = (M.T * M).I * M.T * b
        """
        h = (self.M.T * self.M).I * self.M.T * self.b
        h = np.vstack((h, 
                       np.ones((1, 1))))
        H = h.reshape((3, 3))
        return H

    def solve_H_QR(self):
        """
        Solve H by QR.
            M * h = b
            h = R.I * Q.T * b;
        """
        h = np.zeros((8, 1))
        [Q, R] = np.linalg.qr(self.M)
        h = R.I * Q.T * self.b

        h = np.vstack((h, 
                       np.ones((1, 1))))
        H = h.reshape((3, 3))
        return H

    def solve_H_SVD(self):
        """
        Solve H by SVD.
            M * h = b
            A * x = [M | -b] * [h.T | 1].T = 0
            A = U * S * V.T
            x = V[:, -1]
        """
        A = np.hstack((self.M, -self.b))
        [U, E, VT] = np.linalg.svd(A)
        x = VT[-1, :] # V[:, -1] = V.T[-1, :], V的特征列向量即VT特征行向量
        H = x.reshape((3, 3))
        return H

    def get_H_matrix(self, method="OLS"):

        if method == "OLS":
            self.matrix = self.solve_H_OLS()
        elif method == "QR":
            self.matrix = self.solve_H_QR()
        elif method == "SVD":
            self.matrix = self.solve_H_SVD()
        return self.matrix
