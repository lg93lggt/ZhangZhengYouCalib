import numpy as np

from homography import *

class Calibrator(object):
    """description of class"""

    def __init__(self, chessboard_shape, image_num):
        self.chessboard_shape = chessboard_shape
        self.points_num = chessboard_shape[0] * chessboard_shape[1]
        self.real_points_homogeneous_list = []
        self.image_points_homogeneous_list = []
        self.image_num = image_num
        self.H_list = []
        self.V = np.zeros((2 * self.image_num, 6))
        return

    #def normalize_data(self, data_homogeneous):

    #    avg_x = np.mean(data_homogeneous[:, :, 0])
    #    avg_y = np.mean(data_homogeneous[:, :, 1])
    #    sx = np.sqrt(2) / np.std(data_homogeneous[:, :, 0])
    #    sy = np.sqrt(2) / np.std(data_homogeneous[:, :, 1])

    #    norm_matrix = np.matrix([[sx,  0, -sx * avg_x],
    #                             [ 0, sy, -sy * avg_y],
    #                             [ 0,  0,           1]])
    #    return

    def append_data(self, real_points_homogeneous, image_points_homogeneous):
        self.real_points_homogeneous_list.append(real_points_homogeneous)
        self.image_points_homogeneous_list.append(image_points_homogeneous)
        return

    def init_v_ij(self, H, i, j):
        v_ij = np.array([H[i, 0]*H[j, 0],
                         H[i, 0]*H[j, 1] + H[i, 1]*H[j, 0],
                         H[i, 0]*H[j, 2] + H[i, 2]*H[j, 0],
                         H[i, 1]*H[i, 1],
                         H[i, 1]*H[j, 2] + H[i, 2]*H[j, 1],
                         H[i, 2]*H[j, 2]])
        return v_ij

    def init_V(self):
        V = np.zeros((2 * self.image_num, 6))
        for i in range(self.image_num):
            V[2 * i, :] = self.init_v_ij(self.H_list[i], 0, 1) # v_01
            V[2 * i + 1, :] = self.init_v_ij(self.H_list[i], 0, 0) - self.init_v_ij(self.H_list[i], 1, 1) # v_00 - v_11
        return V

    def solve_b(self, V):
        """
        Vb = 0
        solve b
        b = [B00  B01 B02 B11 B12 B22]
               0    1   2   3   4   5
        """
        U, S, VT = np.linalg.svd(V)
        b = VT[:, -1]
        return b

    def get_intrinsic(self, b):

        cy = (b[1]*b[2] - b[0]*b[4]) / (b[0]*b[3] - b[1]**2)
        k = b[5] - (b[2]**2 + cy*(b[1]*b[2] - b[0]*b[4])) / b[0]
        fx = np.sqrt(k / b[0])
        fy = np.sqrt(k*b[0] / (b[0]*b[3] - b[1]**2))
        c = - b[1] * (fx**2) * fy / k
        cx = c * cy / fx - b[2] * (fx**2) / k
        self.intrinsic = np.array([[fx, c, cx],
                                   [0, fy, cy],
                                   [0,  0,  1]])
        return

    def get_extrinsic(self):
        """

        """
        #for H in self.H_list:
        #    h0 = H[:, 0]

        #    h1 = H[:, 1]

        #    h2 = H[:, 2]

        #    f = 1 / np.linalg.norm(np.dot(self.intrinsic.I, h0))

        #    r0 = scale_factor * np.dot(self.intrinsic.I, h0)

        #    r1 = scale_factor * np.dot(self.intrinsic.I, h1)

        #    t = scale_factor * np.dot(self.intrinsic.I, h2)

        #    r2 = np.cross(r0, r1)

        #    R = np.array([r0, r1, r2, t]).transpose()

        #    extrinsics_param.append(R)

        return

    def calibrate(self):
        for i in range(self.image_num):
            real_points_homogeneous = self.real_points_homogeneous_list[i]
            image_points_homogeneous = self.image_points_homogeneous_list[i]
            H = Homography(real_points_homogeneous, image_points_homogeneous)
            self.H_list.append(H.get_H_matrix(method="SVD"))
            h1 = H.get_H_matrix(method="OLS").reshape((9, 1))[0:8]
            h2 = H.get_H_matrix(method="QR").reshape((9, 1))[0:8]
            h3 = H.get_H_matrix(method="SVD").reshape((9, 1))[0:8]
            d1 = H.b - H.M*h1
            d2 = H.b - H.M*h2
            d3 = H.b - H.M*h3
            print((d1.T*d1)[0,0])
            print((d2.T*d2)[0,0])
            print((d3.T*d3)[0,0])

        del(H)
        V = self.init_V()
        b = self.solve_b(V)
        self.get_intrinsic(b)
        self.get_extrinsic()
        print()
        return




