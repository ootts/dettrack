import cv2
import glob
import argparse
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy import ndimage
import os
from disprcnn.utils.plt_utils import stereo_images_grid

left = cv2.imread('data/img1.jpg', cv2.IMREAD_UNCHANGED)
right = cv2.imread('data/img2.jpg', cv2.IMREAD_UNCHANGED)

# left = (left/256).astype('uint8')
# right = (right/256).astype('uint8')


cameraMatrix1 = np.array([[1485.8503101355045, 0, 641.0072474534551],
                          [0, 1486.8249802291273, 454.1981417235667],
                          [0, 0, 1]])
cameraMatrix2 = np.array([[1472.34425902698, 0, 656.7358738783742],
                          [0, 1473.184475795988, 441.016803589085],
                          [0, 0, 1]])
distCoeffs1 = np.array(
    [-0.09236217303671054, 0.15801009565677457, 0.0020679941868083445, -0.0023435708660260184, 0.04491629603683055])
distCoeffs2 = np.array(
    [-0.09949068652688753, 0.22953391558591676, 0.0016749995113326907, -0.0015940937703328348, -0.13603886268508916])
rotationMatrix = np.array([[0.9999169807005986, 0.0026862926847088424, -0.012602203704541104],
                           [-0.002633967055223802, 0.9999878496600472, 0.0041668633079119935],
                           [0.012613243997904163, -0.004133323588458492, 0.9999119069757908]])
transVector = np.array([29.96389633009774, 0.5883268401189343, -5.0370190999346365])

flags = cv2.CALIB_ZERO_DISPARITY
image_size = left.shape[::-1]

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size,
                                                  rotationMatrix, transVector, flags=flags)

leftmapX, leftmapY = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_32FC1)
rightmapX, rightmapY = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_32FC1)

left_remap = cv2.remap(left, leftmapX, leftmapY, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0, 0))
right_remap = cv2.remap(right, leftmapX, rightmapY, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0))
# plt.subplot(1, 2, 1)
# plt.imshow(left, cmap='gray', vmin=0, vmax=255)
# plt.subplot(1, 2, 2)
# plt.imshow(right, cmap='gray', vmin=0, vmax=255)
# plt.show()

stereo_images_grid(left, right)
stereo_images_grid(left_remap, right_remap)
# plt.subplot(1, 2, 1)
# plt.imshow(left_remap, cmap='gray', vmin=0, vmax=255)
# plt.subplot(1, 2, 2)
# plt.imshow(right_remap, cmap='gray', vmin=0, vmax=255)
# plt.show()
#
# plt.title('before')
# tmp = left / 255.0 / 2 + right / 255.0 / 2
# tmp = np.repeat(tmp[:, :, None], 3, axis=-1)
# for line in range(0, int(tmp.shape[0] / 20)):
#     c = COLORS[line % len(COLORS)]
#     tmp[line * 20, :] = np.array(c) / 255.0
#
# plt.imshow(tmp)
# plt.show()
#
# plt.title('after')
# tmp = left_remap / 255.0 / 2 + right_remap / 255.0 / 2
# tmp = np.repeat(tmp[:, :, None], 3, axis=-1)
# for line in range(0, int(tmp.shape[0] / 20)):
#     c = COLORS[line % len(COLORS)]
#     tmp[line * 20, :] = np.array(c) / 255.0
#
# plt.imshow(tmp)
# plt.show()
#
# left_remap_colored = np.repeat(left_remap[:, :, None], 3, axis=-1)
# right_remap_colored = np.repeat(left_remap[:, :, None], 3, axis=-1)
#
# for line in range(0, int(right_remap.shape[0] / 20)):
#     c = COLORS[line % len(COLORS)]
#     left_remap_colored[line * 20, :] = c
#     right_remap_colored[line * 20, :] = c
#
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.axis('off')
# plt.imshow(left_remap_colored)
# plt.subplot(1, 2, 2)
# plt.axis('off')
# # plt.axis('off')
#
# plt.imshow(right_remap_colored)
# # plt.subplot(2, 2, 3)
# # plt.imshow(right_remap, cmap='gray', vmin=0, vmax=255)
# plt.show()
#
# # cv2.namedWindow('output images', cv2.WINDOW_NORMAL)
# cv2.imshow('output images', np.hstack([left_remap, right_remap]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
