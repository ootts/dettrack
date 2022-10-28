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
engine_file = "tmp/yolact_tracking_head-100-fp16.engine"


def infer(engine, x, ref_x):
    evaltime = EvalTime()
    M = x.shape[0]
    N = ref_x.shape[0]
    # total = M * N
    with engine.create_execution_context() as context:
        streams = []
        output_buffers = []
        for i in range(M):
            for j in range(N):
                evaltime('for loop begin')
                bindings = []
                input_image = torch.stack([x[i], ref_x[j]], 0)
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
        for stream in streams:
            stream.synchronize()
    # match_scores = torch.stack(output_buffers).reshape(M, N)
    # dummy = torch.zeros([M, 1]).float().to(match_scores.device)
    # match_scores = torch.cat([dummy, match_scores], dim=-1)


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def main():
    x, ref_x = torch.load("tmp/x_ref_x.pth")
    N = 1
    N = 10000
    with load_engine(engine_file) as engine:
        for _ in range(N):
            infer(engine, x, ref_x)


if __name__ == '__main__':
    main()
