import pdb
import time

import torch
import os.path as osp
import cv2
import numpy as np
import pycuda.driver as cuda

import pycuda.autoinit
import tensorrt as trt

from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.modeling.models.yolact.layers.box_utils import decode
from disprcnn.trt.pointpillars_part2_inference import PointPillarsPart2Inference
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import bind_array_to_input, bind_array_to_output, load_engine


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
    cfg = setup(args)

    spatial_features = torch.load('tmp/spatial_features.pth', 'cuda')
    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "pointpillars_part2.engine")
    if cfg.trt.convert_to_trt.fp16:
        engine_file = engine_file.replace(".engine", "-fp16.engine")
    inferencer = PointPillarsPart2Inference(engine_file)

    cuda_outputs = inferencer.infer(spatial_features)
    inferencer.destory()

    preds_dict = torch.load('tmp/preds_dict.pth', 'cuda')
    box_preds_r, cls_preds_r, dir_cls_preds_r = preds_dict['box_preds'], preds_dict['cls_preds'], preds_dict[
        'dir_cls_preds']
    box_preds = cuda_outputs['box_preds']
    cls_preds = cuda_outputs['cls_preds']
    dir_cls_preds = cuda_outputs['dir_cls_preds']
    print((box_preds_r - box_preds).abs().max())
    print((cls_preds_r - cls_preds).abs().max())
    print((dir_cls_preds - dir_cls_preds).abs().max())


if __name__ == '__main__':
    main()
