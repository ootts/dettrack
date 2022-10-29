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

    def infer(self, voxels, num_points, coordinates):
        self.context.set_binding_shape(self.engine.get_binding_index("voxels"), (voxels.shape[0], 100, 4))
        self.context.set_binding_shape(self.engine.get_binding_index("num_points"), (num_points.shape[0],))
        self.context.set_binding_shape(self.engine.get_binding_index("coordinates"), (coordinates.shape[0], 4))
        self.bindings = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # size = trt.volume(self.engine.get_binding_shape(binding))
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

        np.copyto(host_inputs['voxels'], voxels.ravel())
        np.copyto(host_inputs['num_points'], num_points.ravel())
        np.copyto(host_inputs['coordinates'], coordinates.ravel())
        # np.copyto(host_inputs['anchors'], anchors.ravel())
        for k in host_inputs.keys():
            cuda.memcpy_htod_async(cuda_inputs[k], host_inputs[k], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for k in host_outputs.keys():
            cuda.memcpy_dtoh_async(host_outputs[k], cuda_outputs[k], stream)
        stream.synchronize()
        self.ctx.pop()
        print()

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

    pp_input = torch.load('tmp/pp_input.pth', 'cpu')
    voxels = pp_input['voxels'].numpy()
    num_points = pp_input['num_points'].numpy()
    coordinates = pp_input['coordinates'].numpy()
    anchors = pp_input['anchors'].numpy()

    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "pointpillars.engine")

    inferencer = Inference(engine_file)

    inferencer.infer(voxels, num_points, coordinates)
    inferencer.destory()

    voxel_features_r = torch.load('tmp/voxel_features.pth', 'cpu')
    print(np.abs(voxel_features_r.numpy() - inferencer.host_outputs['output'].reshape(-1, 64)).max())
    # locr, confr, maskr, priorsr, protor = d['loc'], d['conf'], d['mask'], d['priors'], d['proto']
    # loc = torch.from_numpy(inferencer.host_outputs['loc'].reshape(2, 11481, 4))
    # conf = torch.from_numpy(inferencer.host_outputs['conf'].reshape(2, 11481, 2))
    # mask = torch.from_numpy(inferencer.host_outputs['mask'].reshape(2, 11481, 32))
    # priors = torch.from_numpy(inferencer.host_outputs['priors'].reshape(11481, 4))
    # proto = torch.from_numpy(inferencer.host_outputs['proto'].reshape(2, 76, 150, 32))
    # feat_out = torch.from_numpy(inferencer.host_outputs['feat_out'].reshape(2, 256, 38, 75))
    # print((loc[0] - locr).abs().max())
    # print((conf[0] - confr).abs().max())
    # print((mask[0] - maskr).abs().max())
    # print((priors - priorsr).abs().max())
    # print((proto[0] - protor).abs().max())
    # print((feat_out[0] - feat_outr).abs().max())


if __name__ == '__main__':
    main()
