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
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import bind_array_to_input, bind_array_to_output, load_engine, torch_device_from_trt


class Inference:
    def __init__(self, engine_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        # host_inputs = []
        cuda_inputs = []
        # host_outputs = {}
        cuda_outputs = {}
        bindings = []

        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            dtype = torch_dtype_from_trt(engine.get_binding_dtype(binding_idx))
            shape = tuple(engine.get_binding_shape(binding_idx))
            device = torch_device_from_trt(engine.get_location(binding_idx))
            cuda_mem = torch.empty(size=shape, dtype=dtype, device=device)

            # bindings.append(int(cuda_mem))
            bindings.append(int(cuda_mem.data_ptr()))
            if engine.binding_is_input(binding):
                # host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                # host_outputs[binding] = host_mem
                cuda_outputs[binding] = cuda_mem
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        # self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        # self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, detector, input_file1, input_file2):
        self.ctx.push()

        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        # host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        # host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        img1 = preprocess(input_file1)
        img2 = preprocess(input_file2)

        input_image = torch.stack([img1, img2]).cuda()
        # np.copyto(host_inputs[0], input_image.ravel())
        cuda_inputs[0].copy_(input_image)
        # cuda.memcpy_htod_async(cuda_inputs[0], cuda_inputs[0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # for k in host_outputs.keys():
        #     cuda.memcpy_dtoh_async(host_outputs[k], cuda_outputs[k], stream)
        stream.synchronize()
        self.ctx.pop()

    def destory(self):
        self.ctx.pop()


def preprocess(input_file):
    image = cv2.imread(input_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.repeat(image[:, :, None], 3, axis=-1)
    image = image.astype(np.float32)
    img_h, img_w, _ = image.shape
    width, height = 600, 300
    image = cv2.resize(image, (width, height))
    image = image.astype(np.float32)
    mean = np.array([103.94, 116.78, 123.68])
    std = np.array([57.38, 57.12, 58.4])
    image = (image - mean) / std
    image = image[:, :, [2, 1, 0]]
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    image = torch.from_numpy(image).float()
    return image


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

    inferencer = Inference(engine_file)

    inferencer.infer(detector, input_file1, input_file2)
    inferencer.destory()

    d, feat_outr = torch.load('tmp/yolact_out_ref.pth', 'cuda')
    locr, confr, maskr, priorsr, protor = d['loc'], d['conf'], d['mask'], d['priors'], d['proto']
    loc = inferencer.cuda_outputs['loc']
    conf = inferencer.cuda_outputs['conf']
    mask = inferencer.cuda_outputs['mask']
    priors = inferencer.cuda_outputs['priors']
    proto = inferencer.cuda_outputs['proto']
    feat_out = inferencer.cuda_outputs['feat_out']
    print((loc[0] - locr).abs().max())
    print((conf[0] - confr).abs().max())
    print((mask[0] - maskr).abs().max())
    print((priors - priorsr).abs().max())
    print((proto[0] - protor).abs().max())
    print((feat_out[0] - feat_outr).abs().max())


if __name__ == '__main__':
    main()
