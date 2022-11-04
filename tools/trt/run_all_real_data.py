import os

import imageio
import tensorrt as trt
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import os.path as osp
import glob

import yaml
from PIL import Image
from torchvision.ops import RoIAlign

from disprcnn.data import make_data_loader
from disprcnn.engine.defaults import setup
from disprcnn.engine.defaults import default_argument_parser
from disprcnn.modeling.models.pointpillars.submodules import PointPillarsScatter
from disprcnn.modeling.models.psmnet.inference import DisparityMapProcessor
from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.structures.bounding_box_3d import Box3DList
from disprcnn.structures.boxlist_ops import boxlist_iou
from disprcnn.structures.calib import Calib
from disprcnn.trt.idispnet_inference import IDispnetInference
from disprcnn.trt.pointpillars_part1_inference import PointPillarsPart1Inference
from disprcnn.trt.pointpillars_part2_inference import PointPillarsPart2Inference
from disprcnn.trt.total_inference import TotalInference
from disprcnn.trt.yolact_inference import YolactInference
from disprcnn.trt.yolact_tracking_head_inference import YolactTrackingHeadInference
from disprcnn.utils.pn_utils import to_tensor, to_array
from disprcnn.utils.ppp_utils.box_coders import GroundBox3dCoderTorch
from disprcnn.utils.ppp_utils.box_np_ops import sparse_sum_for_anchors_mask, fused_get_anchors_area, \
    rbbox2d_to_near_bbox
from disprcnn.utils.ppp_utils.target_assigner import build_target_assigner
from disprcnn.utils.ppp_utils.voxel_generator import build_voxel_generator
from disprcnn.utils.pytorch_ssim import ssim, SSIM
from disprcnn.utils.stereo_utils import expand_box_to_integer_torch
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.utils_3d import matrix_3x4_to_4x4
from disprcnn.utils.vis3d_ext import Vis3D


def main():
    # modify this.
    raw_dir = "data/real"
    input_file1s = sorted(glob.glob(osp.join(raw_dir, "left/*.png")))
    input_file2s = sorted(glob.glob(osp.join(raw_dir, "right/*.png")))
    assert len(input_file1s) == len(input_file2s)

    inference = TotalInference(raw_dir)

    for i in range(len(input_file1s)):
        # if int(input_file1s[i][:-4][-6:]) >= 7865: # uncomment to skip images without cars.
        inference.infer(input_file1s[i], input_file2s[i])

    inference.destroy()


if __name__ == '__main__':
    main()
