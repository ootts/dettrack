import torch
import os.path as osp
import cv2
import numpy as np
import pycuda.driver as cuda

import pycuda.autoinit
import tensorrt as trt
from torch.fx.experimental.fx2trt import torch_dtype_from_trt

from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.modeling.models.yolact.layers.box_utils import decode
from disprcnn.trt.yolact_inference import YolactInference
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import bind_array_to_input, bind_array_to_output, load_engine, torch_device_from_trt


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
    cfg = setup(args)

    input_file1 = "data/kitti/tracking/training/image_02/0001/000000.png"
    input_file2 = "data/kitti/tracking/training/image_03/0001/000000.png"
    yolact_cfg = cfg.model.yolact
    detector = Detect(yolact_cfg, yolact_cfg.num_classes, bkg_label=0, top_k=yolact_cfg.nms_top_k,
                      conf_thresh=yolact_cfg.nms_conf_thresh, nms_thresh=yolact_cfg.nms_thresh)

    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "yolact.engine")
    if cfg.trt.convert_to_trt.fp16:
        engine_file = engine_file.replace(".engine", "-fp16.engine")

    inferencer = YolactInference(engine_file, detector)

    cuda_outputs = inferencer.infer(input_file1, input_file2)
    inferencer.destory()

    d, feat_outr = torch.load('tmp/yolact_out_ref.pth', 'cuda')
    locr, confr, maskr, priorsr, protor = d['loc'], d['conf'], d['mask'], d['priors'], d['proto']
    loc = cuda_outputs['loc']
    conf = cuda_outputs['conf']
    mask = cuda_outputs['mask']
    priors = cuda_outputs['priors']
    proto = cuda_outputs['proto']
    feat_out = cuda_outputs['feat_out']
    print((loc[0] - locr).abs().max())
    print((conf[0] - confr).abs().max())
    print((mask[0] - maskr).abs().max())
    print((priors - priorsr).abs().max())
    print((proto[0] - protor).abs().max())
    print((feat_out[0] - feat_outr).abs().max())


if __name__ == '__main__':
    main()
