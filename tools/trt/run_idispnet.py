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
engine_file = "tmp/idispnet-100.engine"


def infer(engine, left, right):
    input_image = np.stack([left, right])
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


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def main():
    left_roi_images, right_roi_images = torch.load('tmp/left_right_roi_images.pth')
    left_roi_images = left_roi_images.cpu().numpy()
    right_roi_images = right_roi_images.cpu().numpy()
    with load_engine(engine_file) as engine:
        for _ in range(10000):
            infer(engine, left_roi_images, right_roi_images)


if __name__ == '__main__':
    main()
