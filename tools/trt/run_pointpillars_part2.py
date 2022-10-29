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
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import bind_array_to_input, bind_array_to_output, load_engine


class Inference:
    def __init__(self, engine_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs = {}
        cuda_inputs = {}
        host_outputs = {}
        cuda_outputs = {}
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs[binding] = host_mem
                cuda_inputs[binding] = cuda_mem
            else:
                host_outputs[binding] = host_mem
                cuda_outputs[binding] = cuda_mem
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, spatial_features):
        self.ctx.push()

        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        np.copyto(host_inputs['input'], spatial_features.ravel())
        for k in host_inputs.keys():
            cuda.memcpy_htod_async(cuda_inputs[k], host_inputs[k], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for k in host_outputs.keys():
            cuda.memcpy_dtoh_async(host_outputs[k], cuda_outputs[k], stream)
        stream.synchronize()
        self.ctx.pop()

    def destory(self):
        self.ctx.pop()


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
    cfg = setup(args)

    spatial_features = torch.load('tmp/spatial_features.pth', 'cpu').numpy()
    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "pointpillars_part2.engine")

    inferencer = Inference(engine_file)

    inferencer.infer(spatial_features)
    inferencer.destory()

    preds_dict = torch.load('tmp/preds_dict.pth', 'cpu')
    box_preds_r, cls_preds_r, dir_cls_preds_r = preds_dict['box_preds'], preds_dict['cls_preds'], preds_dict[
        'dir_cls_preds']
    box_preds = inferencer.host_outputs['box_preds'].reshape(1, 248, 216, 14)
    cls_preds = inferencer.host_outputs['cls_preds'].reshape(1, 248, 216, 2)
    dir_cls_preds = inferencer.host_outputs['dir_cls_preds'].reshape(1, 248, 216, 4)
    print((box_preds_r - torch.from_numpy(box_preds)).abs().max())
    print((cls_preds_r - torch.from_numpy(cls_preds)).abs().max())
    print((dir_cls_preds - torch.from_numpy(dir_cls_preds)).abs().max())


if __name__ == '__main__':
    main()
