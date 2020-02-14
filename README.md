# ZhangZhengYouCalib
相机标定常用Matlab工具箱或OpenCV标定.详细原理如下:

1.相机成像模型
记世界坐标为, 像素坐标为.
齐次坐标分别为, .
建立小孔成像模型:
       (1)
其中, s为任意比例因子, A为相机内参, R为3x3的旋转矩阵, t为3x1的平移矩阵.
fx, fy分别为图像在u, v轴的比例因子. cx, cy为偏移量, 为径向畸变参数.

2.标定模型平面与图像单应性关系
准备并拍摄标定板.获取标定板上的人工特征点,本例为棋盘格角点.
设标定板平面上点的世界坐标中Z为0.则式1中:

此时,, . 此时可以用3x3的单应矩阵H表示:
                    (2)

3.求解单应性矩阵H
H[3, 3]=1, 此时:
                 (3)

即:                     



H11-h32一共8个自由度, 故需4组线性无关点, 使得H有解:

而在实际情况中, 由于噪声与测量误差的存在, 棋盘格角点同行列上点并非线性相关,故对于有m x n组特征点对时, 有如下超定方程组:
 
记为, 记.

A.封闭解
即求, 损失函数
导数时, 最小二乘解为.
B.QR
C.SVD

4.内参的约束条件
由(2)(3)式可知:

, 
又由旋转矩阵特性可知:
r1, r2正交, 即:                               (4)
|r1|=|r2|, 即:               (5)

5.相机标定解
A.封闭解
 (6)
式(6)中B为对称矩阵, 定义向量
记H中第i列向量为, 则由式(6)得:
  (7)
由式(4), (5), (7)得:
	                            (8)
若观测k张图片(k>=3), 则由式(8)可得方程组:
                                   (9)

6.求解内参矩阵A

7.求解外参矩阵

8.求解畸变参数

9.非线性优化



