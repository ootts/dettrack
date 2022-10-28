import torch
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import os.path as osp
import glob
import os

from torch.fx.experimental.fx2trt import torch_dtype_from_trt

from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import load_engine, torch_device_from_trt


class Predictor:
    def __init__(self):
        from disprcnn.engine.defaults import setup
        from disprcnn.engine.defaults import default_argument_parser
        parser = default_argument_parser()
        args = parser.parse_args()
        args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
        cfg = setup(args)

        yolact_cfg = cfg.model.yolact
        self.detector = Detect(yolact_cfg, yolact_cfg.num_classes, bkg_label=0, top_k=yolact_cfg.nms_top_k,
                               conf_thresh=yolact_cfg.nms_conf_thresh, nms_thresh=yolact_cfg.nms_thresh)
        self.yolact_engine_file = "tmp/yolact-100.engine"
        self.yolact_tracking_engine_file = "tmp/yolact_tracking_head-100-fp16.engine"
        self.idispnet_engine_file = "tmp/idispnet-100.engine"

    def inference(self, input_file1s, input_file2s):
        with load_engine(self.yolact_engine_file) as yolact_engine, \
                load_engine(self.yolact_tracking_engine_file) as yolact_tracking_engine, \
                load_engine(self.idispnet_engine_file) as idispnet_engine:
            for i in range(len(input_file1s)):
                input_file1 = input_file1s[i]
                input_file2 = input_file2s[i]
                self.infer(yolact_engine, self.detector, yolact_tracking_engine, idispnet_engine,
                           input_file1, input_file2)

    def infer(self, yolact_engine, detector, yolact_tracking_engine, idispnet_engine, input_file1, input_file2):
        pass

    def infer_yolact(self, yolact_engine, detector, input_file1, input_file2):
        img1 = self.preprocess(input_file1)
        img2 = self.preprocess(input_file2)
        input_image = torch.stack([img1, img2])
        evaltime = EvalTime()
        with yolact_engine.create_execution_context() as context:
            bindings = []
            for binding in yolact_engine:
                binding_idx = yolact_engine.get_binding_index(binding)
                # size = trt.volume(context.get_binding_shape(binding_idx))
                # dtype = trt.nptype(engine.get_binding_dtype(binding))
                if yolact_engine.binding_is_input(binding):
                    inputs_torch = input_image.to(torch_device_from_trt(yolact_engine.get_location(binding_idx)))
                    inputs_torch = inputs_torch.type(torch_dtype_from_trt(yolact_engine.get_binding_dtype(binding_idx)))
                    bindings.append(int(inputs_torch.data_ptr()))
                else:
                    dtype = torch_dtype_from_trt(yolact_engine.get_binding_dtype(binding_idx))
                    shape = tuple(yolact_engine.get_binding_shape(binding_idx))
                    device = torch_device_from_trt(yolact_engine.get_location(binding_idx))
                    output = torch.empty(size=shape, dtype=dtype, device=device)
                    bindings.append(int(output.data_ptr()))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            # Run inference
            evaltime('trt begin')
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            evaltime('trt end')
            # Transfer prediction output from the GPU.
            # Synchronize the stream
            stream.synchronize()
        unified_out = output.reshape(2, 11481, -1)
        loc = unified_out[:, :, :4]
        conf = unified_out[:, :, 4:4 + 2]
        mask = unified_out[:, :, 4 + 2:4 + 2 + 32]
        prior = unified_out[:, :, 4 + 2 + 32:4 + 2 + 32 + 4][0]
        proto = unified_out[:, :11400, 4 + 2 + 32 + 4:4 + 2 + 32 + 4 + 32].reshape(2, 76, 150, 32)
        feat_out = unified_out[:, :9728, 4 + 2 + 32 + 4 + 32:].reshape(2, 256, 38, 75)

        pred_outs = {'loc': loc,
                     'conf': conf,
                     'mask': mask,
                     'priors': prior,
                     'proto': proto}
        dets_2d = detector(pred_outs)
        evaltime('decode')
        return dets_2d

    def preprocess(self, input_file):
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
        image = torch.from_numpy(image).cuda()
        return image


def main():
    data2_dir = "data/kitti/tracking/training/image_02/0001/"
    data3_dir = "data/kitti/tracking/training/image_03/0001/"
    input_file1s = sorted(glob.glob(osp.join(data2_dir, "*.png")))
    input_file2s = sorted(glob.glob(osp.join(data3_dir, "*.png")))
    assert len(input_file1s) == len(input_file2s)
    predictor = Predictor()
    predictor.inference(input_file1s, input_file2s)


if __name__ == '__main__':
    main()
