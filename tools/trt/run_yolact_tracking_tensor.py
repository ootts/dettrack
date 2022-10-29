import torch
import os.path as osp
import cv2
import numpy as np
import pycuda.driver as cuda

import pycuda.autoinit
import tensorrt as trt

from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.modeling.models.yolact.layers.box_utils import decode
from disprcnn.trt.yolact_tracking_head_inference import YolactTrackingHeadInference
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

    x, ref_x = torch.load('tmp/x_ref_x.pth', 'cuda')

    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "yolact_tracking_head.engine")

    inferencer = YolactTrackingHeadInference(engine_file)

    inferencer.infer(x, ref_x)
    inferencer.destory()

    # match_score = torch.load('tmp/match_score.pth', 'cuda')
    # print((match_score[0][0] - inferencer.cuda_outputs['output']).abs().max())


if __name__ == '__main__':
    main()
