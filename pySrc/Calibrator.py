import numpy as np
import scipy
import cv2
import glob

from Homography import *

class Calibrator(object):
    """
    Camera Calibrator.
    Usage:
    1. Calibrator()
    1. ret = Calibrator.init_data()
    2. if ret:
           Calibrator.calibrate()
    """
    def __init__(self, chessboard_edge_length, chessboard_shape, input_dir, method="QR" ):
        """
        Argin:
            chessboard_edge_length: length of chessboard's square edge (mm).
            chessboard_shape: a string which means shape of chessboard inner points, eg:"7x11", at least "2x2".
            input_dir: a string of input image folder directory.
            method: optional, use "QR"/"OLS"/"SVD" to solve equation, default="QR".
        """
        self.image_folder_dir = input_dir
        self.image_dir_list = glob.glob(input_dir + "/*.*")
        self.image_num = len(self.image_dir_list)

        self.chessboard_shape = chessboard_shape
        self.chessboard_points_num = chessboard_shape[0] * chessboard_shape[1]

        self.world_points_list = []
        self.image_points_list = []
        self.H_list = []
        return

    def get_normalize_matrix(self, input_data):
        """
        Get a (3, 3) normalized matrix of input data.
        """
        x_avg = np.mean(input_data[:, 0])
        y_avg = np.mean(input_data[:, 1])
        sx = np.sqrt(2) / np.std(input_data[:, 0])
        sy = np.sqrt(2) / np.std(input_data[:, 1])

        norm_matrix = np.matrix([[sx,  0, - sx * x_avg],
                                 [ 0, sy, - sy * y_avg],
                                 [ 0,  0,            1]])
        return norm_matrix

    def normalize_data(self):
        """
        Normalize world points and image points for all images.
        """
        self.world_points_norm_list = []
        self.image_points_norm_list = []
        self.world_norms_list = []
        self.image_norms_list = []
        for i in range(self.image_num):
            world_points_norm = np.zeros((self.chessboard_points_num, 3))
            image_points_norm = np.zeros((self.chessboard_points_num, 3))

            world_norm_matrix = self.get_normalize_matrix(self.world_points_list[i])
            image_norm_matrix = self.get_normalize_matrix(self.image_points_list[i])

            world_points_homo = self.to_homogeneous(self.world_points_list[i])
            image_points_homo = self.to_homogeneous(self.image_points_list[i])

            for p in range(self.chessboard_points_num): # 简化:(norm_matrix * p.T).T = p * norm_matrix.T
                world_points_norm[p, :] = np.array(np.matrix(world_points_homo[p]) * world_norm_matrix.T)
                image_points_norm[p, :] = np.array(np.matrix(image_points_homo[p]) * image_norm_matrix.T)
            self.world_points_norm_list.append(world_points_norm.copy())
            self.image_points_norm_list.append(image_points_norm.copy())
            self.image_norms_list.append(image_norm_matrix.copy())
            self.world_norms_list.append(world_norm_matrix.copy())
        return

    def init_data(self): 
        """
        Init data from all images.
        """
        world_points = np.mgrid[0:self.chessboard_shape[0], 0:self.chessboard_shape[1]].T.reshape(self.chessboard_points_num, 2)
        for [image_index, image_dir] in enumerate(self.image_dir_list):
            image = cv2.imread(image_dir)
            is_find, chessboard_corners = cv2.findChessboardCorners(image, self.chessboard_shape, None) # 角点顺序先列后行!!!
            if is_find:
                image_points = chessboard_corners.reshape(self.chessboard_points_num, 2)
                self.image_points_list.append(image_points.copy()) # 坐标值后续计算会改变, 需要深拷贝
                self.world_points_list.append(world_points.copy())
                #if IS_OUTPUT:
                #    cv2.drawChessboardCorners(image, chessboard_shape, chessboard_corners, True)
                #    cv2.imwrite()
            else:
                print("ERR in:", image_dir)
        print("Init data SUCC.")
        return

    def to_homogeneous(self, input_points):
        """
        Transform input array or matrix to homogeneous.
        """
        shape = input_points.shape
        if len(shape) == 1:
            output_points = np.append(input_points, 1)
        elif len(shape) > 1:
            tmp = np.ones((input_points.shape[0], 1))
            output_points = np.hstack((input_points, tmp))
        return output_points

    def init_v_ij(self, H, i, j):
        """
        Init v_ij array.
        cv2.findChessboardCorners() 顺序先列后行, 函数内索引顺序要一致
        """
        v_ij = np.array([H[0, i] * H[0, j],
                         H[0, i] * H[1, j] + H[1, i]*H[0, j],
                         H[0, i] * H[2, j] + H[2, i]*H[0, j],
                         H[1, i] * H[1, j],
                         H[1, i] * H[2, j] + H[2, i]*H[1, j],
                         H[2, i] * H[2, j]])
        return v_ij

    def init_V(self):
        """
        Init V matrix.
        A (2*image_num, 6) V matrix for all images.
        """
        V = np.zeros((2 * self.image_num, 6))
        for i in range(self.image_num):
            V[2 * i, :] = self.init_v_ij(self.H_list[i], 0, 1) # v_01
            V[2 * i + 1, :] = self.init_v_ij(self.H_list[i], 0, 0) - self.init_v_ij(self.H_list[i], 1, 1) # v_00 - v_11
        self.V = V
        return

    def solve_b(self):
        """
        Vb = 0
        solve b
        b = [B00  B01 B02 B11 B12 B22]
                0    1   2   3   4   5
        """
        V = self.V
        U, S, VT = np.linalg.svd(V)
        b = VT[-1, :]
        self.b = b
        return

    def get_intrinsic(self):
        """
        Get intrinsic matrix.
        A (3, 3) intrinsic matrix for all images.
        """
        b = self.b

        cy = (b[1] * b[2] - b[0] * b[4]) / (b[0] * b[3] - b[1]**2)
        k_square = b[5] - (b[2]**2 + cy * (b[1] * b[2] - b[0] * b[4])) / b[0]
        fx = np.sqrt(k_square / b[0])
        fy = np.sqrt(k_square * b[0] / (b[0] * b[3] - b[1]**2))
        c = - b[1] * (fx**2) * fy / k_square
        cx = c * cy / fx - b[2] * (fx**2) / k_square
        self.intrinsic = np.matrix([[fx, c, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        return

    def get_extrinsic(self):
        """
        Get extrinsic matrix list.
        A (3, 3) extrinsic matrix in list for a single image.
        """
        self.extrinsices_list = []
        for H in self.H_list:
            scale_factor = 2 / (np.linalg.norm(self.intrinsic.I * H[:, 0]) + np.linalg.norm(self.intrinsic.I * H[:, 1])) # A.I*h0和A.I*h1的模取平均
            r0 = scale_factor * self.intrinsic.I * H[:, 0]
            r1 = scale_factor * self.intrinsic.I * H[:, 1]
            r2 = np.cross(r0, r1, axis=0) # 需定义叉乘方向axis
            t  = scale_factor * self.intrinsic.I * H[:, 2]
            Rt = np.hstack((r0, r1, r2, t))
            self.extrinsices_list.append(Rt)
        return

    def get_distortion(self):
        """
        Get distortion array.
        [k1 k2 p1 p2 k3]
        A (5, ) intrinsic matrix for all images.
        p1, p2, k3 = 0.
        """
        cx = self.intrinsic[0, 2]
        cy = self.intrinsic[1, 2]

        D = np.zeros((2 * self.image_num * self.chessboard_points_num, 2))
        d = np.zeros((2 * self.image_num * self.chessboard_points_num, 1))

        # init matrix D, d.
        for i in range(self.image_num):
            for j in range(self.chessboard_points_num):
                index = i * self.chessboard_points_num + j

                M = self.world_points_list[i][j]
                M = np.append(M, [0, 1])
                RtM = self.extrinsices_list[i] * np.mat(M).T
                RtM /= RtM[-1, 0]

                r_square = (RtM[0, 0]** 2 + RtM[1, 0]**2)

                m_estimate = self.intrinsic * RtM
                m_estimate  /= m_estimate [-1, 0]

                m_observed = np.mat(self.to_homogeneous(self.image_points_list[i][j])).T

                D[index,     0:2] = (m_observed[0, 0] - cx) * np.array([r_square, r_square**2]) # (u -cx)r^2, (u -cx)r^4
                D[index + 1, 0:2] = (m_observed[1, 0] - cy) * np.array([r_square, r_square**2]) # (v -cy)r^2, (v -cy)r^4

                d[index,     0] = m_estimate[0, 0] - m_observed[0, 0] # u_ - u
                d[index + 1, 0] = m_estimate[1, 0] - m_observed[1, 0] # v_ - v
        D = np.mat(D)
        d = np.mat(d)
        k = (D.T * D).I * D.T * d
        self.distortion = np.array([k[0, 0], k[1, 0], 0, 0, 0]) # k1 k2 p1 p2 p3
        return

    def compose_params(self):
        """
        Compose all parameters in an (7 + 12*image_num) array P.
        P[0:7] = [fx, fy, c, cx, cy, k1, k2]
        P[7:7 + 12 * image_index] = [r1.T, r2.T, r3.T, t.T]
        Or (7 + 6*image_num) array when use cv2.Rodrigues to compress R = [r1, r2, r3] martrix.
        """
        P = np.zeros((7 + self.image_num * 12))

        fx = self.intrinsic[0, 0]
        fy = self.intrinsic[1, 1]
        c  = self.intrinsic[0, 1]
        cx = self.intrinsic[0, 2]
        cy = self.intrinsic[1, 2]
        k1 = self.distortion[0]
        k2 = self.distortion[1]
        P[0 : 7] = np.array([fx, fy, c, cx, cy, k1, k2])
        for i in range(self.image_num):
            index = 7 + 12 * i
            tmp = np.array(self.extrinsices_list[i].reshape((12, )))
            P[index : index + 12] = tmp.copy()
        return P

    def decompose_params(self, input_params):
        """
        Decompose array P to get intrinsic marix, extrinsic matrix list and distortion array.
        """
        P = input_params
        fx = P[0]
        fy = P[1]
        c  = P[2]
        cx = P[3]
        cy = P[4]
        k1 = P[5]
        k2 = P[6]
        intrinsic = np.matrix([[fx, c, cx],
                               [0, fy, cy],
                               [0,  0,  1]])
        distortion = np.array([k1, k2])
        extrinsices_list = []
        for i in range(self.image_num):
            index = 7 + 12 * i
            extrinsices_list.append(np.mat(P[index : index +12].reshape((3, 4))))
        return intrinsic, extrinsices_list, distortion

    def get_projected_point(self, A, Rt, k, world_point_homo):
        """
        Project homogeneous world point to homogeneous image point.
        """
        M = world_point_homo
        RtM = Rt * np.mat(M).T
        RtM /= RtM[-1, 0]

        r_square = (RtM[0, 0]** 2 + RtM[1, 0]**2)

        image_point_homo = A * RtM
        image_point_homo  /= image_point_homo [-1, 0]
        image_point_homo[0:2, 0] *= (1 + k[0] * r_square + k[1] * r_square**2)
        return image_point_homo

    def object_function(self, params):
        """
        Object function to minimize.
        """
        [A, Rt_list, k] = self.decompose_params(params)
        loss_array = np.zeros((self.image_num, ))
        for i in range(self.image_num):
            error = 0
            for j in range(self.chessboard_points_num):
                index = i * self.chessboard_points_num + j

                M = self.world_points_list[i][j]
                M = np.append(M, [0, 1])

                m_estimate = self.get_projected_point(A, Rt_list[i], k, M)
                m_observed = np.mat(self.to_homogeneous(self.image_points_list[i][j])).T

                error += ((m_observed - m_estimate).T * (m_observed - m_estimate))[0, 0]
            loss_array[i] = error / self.chessboard_points_num

        total_loss = np.sum(loss_array)
        print("Total Loss =", total_loss)
        print("Max   Loss =", np.sqrt(np.max(loss_array)))
        return loss_array

    def get_jacobian(self, input_params):
        [A, Rt_list, k] = self.decompose_params(input_params)
        for i in range(self.image_num):
            error = 0
            for j in range(self.chessboard_points_num):
                index = i * self.chessboard_points_num + j

                M = self.world_points_list[i][j]
                M = np.append(M, [0, 1])

                m_estimate = self.get_projected_point(A, Rt_list[i], k, M)
                m_observed = np.mat(self.to_homogeneous(self.image_points_list[i][j])).T
                error += ((m_observed - m_estimate).T * (m_observed - m_estimate))[0, 0]
            error_array[i] = error / self.chessboard_points_num

        J = np.zeros((K, 2 * M * N))

        for k in range(K):

            J[k] = np.gradient(res, P[k])
        return J.T

    def optimizer(self):
        """
        Minimize object function with init parameters to get refined parameters.
        """
        init_params = self.compose_params()
        self.object_function(init_params)
        out= scipy.optimize.least_squares(fun=self.object_function,
                                           x0=init_params)
                                           #method="lm")
        print(out)
        return

    def calibrate(self):
        """
        Caculate all camera parameters.
        """
        self.normalize_data()
        for i in range(self.image_num):
            world_points_norm = self.world_points_norm_list[i]
            image_points_norm = self.image_points_norm_list[i]
            H = Homography(world_points_norm, image_points_norm)
            H.get_H_matrix(method="QR" )
            H_matrix = self.image_norms_list[i].I * H.matrix * self.world_norms_list[i]
            H_matrix /= H_matrix[-1, -1]
            self.H_list.append(H_matrix)

        del(H)
        self.init_V()
        self.solve_b()
        self.get_intrinsic()
        self.get_extrinsic()
        self.get_distortion()
        self.optimizer()
        print("\nIntrinsic matirx:\n", self.intrinsic)
        for i in range(self.image_num):
            print("\nEntrinsic matirx {:d}:\n".format(i), self.intrinsic)
        return
