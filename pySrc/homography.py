import numpy as np
from scipy.optimize import leastsq

class Homography(object):
    """description of class"""

    def __init__(self, real_points, image_points):
        points_num = real_points.shape[0] * real_points.shape[1]
        M = np.zeros((2 * points_num, 8))
        b = np.zeros((2 * points_num, 1))

        for i in range(real_points.shape[0]):
            for j in range(real_points.shape[1]):
                index1 = 2 * (i * real_points.shape[1] + j)
                index2 = index1 + 1

                M[index1, 0] = real_points[i, j, 0]
                M[index1, 1] = real_points[i, j, 1]
                M[index1, 2] = 1

                M[index2, 3] = real_points[i, j, 0]
                M[index2, 4] = real_points[i, j, 1]
                M[index2, 5] = 1
                
                M[index1 : index2 + 1, 6] = -(real_points[i, j, 0] * image_points[i, j, 0])
                M[index1 : index2 + 1, 7] = -(real_points[i, j, 1] * image_points[i, j, 1])

                b[index1, :] = image_points[i, j, 0]
                b[index2, :] = image_points[i, j, 1]

        self.M = np.matrix(M)
        self.b = np.matrix(b)
        del(M, b)
        return

    def solve_H_OLS(self):
        """
        Solve H by Ordinary Least Square
            M * h = b
            argmin(b - M * h)
            h = (M.T * M).I * M.T * b
        
        """
        
        h = np.zeros((8, 1))
        h = (self.M.T * self.M).I * self.M.T * self.b

        h = np.vstack((h, 
                       np.ones((1, 1))))
        H = h.reshape((3, 3))

        return H

    def solve_H_QR(self):
        """
        Solve H by QR
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
        Solve H by SVD
            M * h = b
            A * x = [M | -b] * [h.T | 1].T = 0
            A = U * S * VT
            x = VT[:, -1]
        """
        h = np.zeros((9, 1))
        A = np.hstack((self.M, -self.b))
        [U, E, VT] = np.linalg.svd(A)
        x = VT[:, -1]
        H = x.reshape((3, 3))
        H /= H[2, 2]
        return H

    def get_H_matrix(self, method="OLS"):

        if method == "OLS":
            self.matrix = self.solve_H_OLS()
        elif method == "QR":
            self.matrix = self.solve_H_QR()
        elif method == "SVD":
            self.matrix = self.solve_H_SVD()
        return self.matrix




