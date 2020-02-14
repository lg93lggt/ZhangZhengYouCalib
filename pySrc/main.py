import numpy as np
import cv2
import gflags
import glob
import os
import sys

from homography import *
from Calibrator import *


if __name__ == "__main__":
    gflags.DEFINE_bool("output", False, "need to output the image_points image?")
    gflags.FLAGS(sys.argv)

    # 标定所用图像
    folder_dir = "C:/Users/Li/work/LTT/calibDataset/L"
    image_dir_list = glob.glob(folder_dir + "/*.*")
    image_number = len(image_dir_list)

    chessboard_shape = (7, 11)

    tmp = np.mgrid[0:chessboard_shape[0] , 0:chessboard_shape[1]]
    real_coordinate_homogeneous = np.zeros((chessboard_shape[0], chessboard_shape[1], 3))
    real_coordinate_homogeneous[:, :, 0] = tmp[0]
    real_coordinate_homogeneous[:, :, 1] = tmp[1]
    real_coordinate_homogeneous[:, :, 2] = 1
    del(tmp)

    real_points_homogeneous  = np.zeros((chessboard_shape[0], chessboard_shape[1], 3))
    image_points_homogeneous = np.zeros((chessboard_shape[0], chessboard_shape[1], 3))

    calibrator = Calibrator(chessboard_shape=chessboard_shape, image_num=len(image_dir_list))

    for [image_index, image_dir] in enumerate(image_dir_list):

        image = cv2.imread(image_dir)

        # 寻找棋盘角点
        is_find, chessboard_corners = cv2.findChessboardCorners(image, chessboard_shape, None)

        if is_find:

            if gflags.FLAGS.output:
                cv2.drawChessboardCorners(image, chessboard_shape, chessboard_corners, True)
                cv2.imwrite()

            real_points_homogeneous[:, :, :]  = real_coordinate_homogeneous
            image_points_homogeneous[:, :, 0:2] = chessboard_corners.reshape(chessboard_shape[0], chessboard_shape[1], 2)
            image_points_homogeneous[:, :, 2] = np.ones(chessboard_shape)
            calibrator.append_data(real_coordinate_homogeneous.copy(), image_points_homogeneous.copy())
        else:
            print("ERR in:", image_dir)
    calibrator.calibrate()
    print()


    #
    #calibrate(real_points, image_points)
