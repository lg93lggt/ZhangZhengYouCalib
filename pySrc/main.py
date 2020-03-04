import sys
import gflags

from Homography import *
from Calibrator import *


if __name__ == "__main__":
    gflags.DEFINE_float("chessboard_edge_length", 1., "chessboard edge's length (mm)")
    gflags.DEFINE_string("chessboard_shape", "", "chessboard inner points' shape")
    gflags.DEFINE_string("input_dir", "", "image folder directory")
    gflags.FLAGS(sys.argv)

    chessboard_shape = tuple(map(int, gflags.FLAGS.chessboard_shape.split("x")))

    calibrator = Calibrator(chessboard_edge_length=gflags.FLAGS.chessboard_edge_length,
                            chessboard_shape=chessboard_shape,
                            input_dir=gflags.FLAGS.input_dir)

    calibrator.init_data()
    calibrator.calibrate()
    print()
