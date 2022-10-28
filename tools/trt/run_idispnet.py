import torch
import os

import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from torch.fx.experimental.fx2trt import torch_dtype_from_trt

from disprcnn.modeling.models.yolact.layers import Detect
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import torch_device_from_trt

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "tmp/idispnet-100.engine"


def infer(engine, left, right):
    """
    :param engine:
    :param left: Nx3x112x112
    :param right: Nx3x112x112
    :return:
    """
    N = left.shape[0]
    evaltime = EvalTime()
    with engine.create_execution_context() as context:
        streams = []
        output_buffers = []
        for i in range(N):
            evaltime('for loop begin')
            bindings = []
            input_image = torch.stack([left[i], right[i]], 0)
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                if engine.binding_is_input(binding):
                    inputs_torch = input_image.to(torch_device_from_trt(engine.get_location(binding_idx)))
                    inputs_torch = inputs_torch.type(torch_dtype_from_trt(engine.get_binding_dtype(binding_idx)))
                    bindings.append(int(inputs_torch.data_ptr()))
                else:
                    dtype = torch_dtype_from_trt(engine.get_binding_dtype(binding_idx))
                    shape = tuple(engine.get_binding_shape(binding_idx))
                    device = torch_device_from_trt(engine.get_location(binding_idx))
                    output = torch.empty(size=shape, dtype=dtype, device=device)
                    output_buffers.append(output)
                    bindings.append(int(output.data_ptr()))

            stream = cuda.Stream()
            streams.append(stream)
            # Transfer input data to the GPU.
            # Run inference
            evaltime('trt begin')
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            evaltime('trt end')
            # Transfer prediction output from the GPU.
        # Synchronize the stream
        for i in range(N):
            streams[i].synchronize()
    print()


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def main():
    left_roi_images, right_roi_images = torch.load('tmp/left_right_roi_images.pth')
    # left_roi_images = left_roi_images.cpu().numpy()
    # right_roi_images = right_roi_images.cpu().numpy()
    with load_engine(engine_file) as engine:
        for _ in range(10000):
            infer(engine, left_roi_images, right_roi_images)


if __name__ == '__main__':
    main()
