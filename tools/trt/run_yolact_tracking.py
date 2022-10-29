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

        # for binding in engine:
        #     size = trt.volume(engine.get_binding_shape(binding))
        #     host_mem = cuda.pagelocked_empty(size, np.float32)
        #     cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        #
        #     bindings.append(int(cuda_mem))
        #     if engine.binding_is_input(binding):
        #         host_inputs[binding] = host_mem
        #         cuda_inputs[binding] = cuda_mem
        #     else:
        #         host_outputs[binding] = host_mem
        #         cuda_outputs[binding] = cuda_mem
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, x, ref_x):
        self.context.set_binding_shape(self.engine.get_binding_index("x"), (x.shape[0], 256, 7, 7))
        self.context.set_binding_shape(self.engine.get_binding_index("ref_x"), (ref_x.shape[0], 256, 7, 7))
        self.bindings = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs[binding] = host_mem
                self.cuda_inputs[binding] = cuda_mem
            else:
                self.host_outputs[binding] = host_mem
                self.cuda_outputs[binding] = cuda_mem

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

        np.copyto(host_inputs['x'], x.ravel())
        np.copyto(host_inputs['ref_x'], ref_x.ravel())
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

    x, ref_x = torch.load('tmp/x_ref_x.pth', 'cpu')
    x = x.numpy()
    ref_x = ref_x.numpy()

    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "yolact_tracking_head.engine")

    inferencer = Inference(engine_file)

    inferencer.infer(x, ref_x)
    inferencer.destory()

    match_score = torch.load('tmp/match_score.pth', 'cpu')
    print(np.abs(
        match_score[0][0].numpy() - inferencer.host_outputs['output'].reshape(x.shape[0], ref_x.shape[0] + 1)).max())


if __name__ == '__main__':
    main()
