import torch
import os

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.utils.timer import EvalTime

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "tmp/yolact.engine"


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def infer(engine, detector, input_file1, input_file2):
    img1 = preprocess(input_file1)
    img2 = preprocess(input_file2)
    input_image = np.stack([img1, img2])
    evaltime = EvalTime()
    # evaltime('begin')
    with engine.create_execution_context() as context:
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        evaltime('begin')
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        evaltime('end')
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
    # evaltime('trt forward')
    # unified_out = output_buffer.reshape(2, 11481, -1)
    # loc = unified_out[:, :, :4]
    # conf = unified_out[:, :, 4:4 + 2]
    # mask = unified_out[:, :, 4 + 2:4 + 2 + 32]
    # prior = unified_out[:, :, 4 + 2 + 32:4 + 2 + 32 + 4][0]
    # proto = unified_out[:, :11400, 4 + 2 + 32 + 4:4 + 2 + 32 + 4 + 32].reshape(2, 76, 150, 32)
    # feat_out = unified_out[:, :9728, 4 + 2 + 32 + 4 + 32:].reshape(2, 256, 38, 75)
    #
    # pred_outs = {'loc': torch.from_numpy(loc).cuda(),
    #              'conf': torch.from_numpy(conf).cuda(),
    #              'mask': torch.from_numpy(mask).cuda(),
    #              'priors': torch.from_numpy(prior).cuda(),
    #              'proto': torch.from_numpy(proto).cuda()}
    # dets_2d = detector(pred_outs)
    # evaltime('decode')
    # print()


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
    return image


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
    cfg = setup(args)

    input_file1 = "data/kitti/tracking/training/image_02/0001/000000.png"
    input_file2 = "data/kitti/tracking/training/image_03/0001/000000.png"
    yolact_cfg = cfg.model.yolact
    detector = Detect(yolact_cfg, yolact_cfg.num_classes, bkg_label=0, top_k=yolact_cfg.nms_top_k,
                      conf_thresh=yolact_cfg.nms_conf_thresh, nms_thresh=yolact_cfg.nms_thresh)

    with load_engine(engine_file) as engine:
        for _ in range(10000):
            infer(engine, detector, input_file1, input_file2)


if __name__ == '__main__':
    main()