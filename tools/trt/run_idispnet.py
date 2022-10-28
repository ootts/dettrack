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
            bindings = []
            input_image = np.stack([left[i], right[i]], axis=0)
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
                    output_buffers.append(output_buffer)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))

            stream = cuda.Stream()
            streams.append(stream)
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            evaltime('')
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            evaltime('trt')
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
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
    left_roi_images = left_roi_images.cpu().numpy()
    right_roi_images = right_roi_images.cpu().numpy()
    with load_engine(engine_file) as engine:
        for _ in range(10000):
            infer(engine, left_roi_images, right_roi_images)


if __name__ == '__main__':
    main()
