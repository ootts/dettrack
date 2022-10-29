import numpy as np
import os.path as osp
import torch
import tensorrt as trt


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


TRT_LOGGER = trt.Logger()


def load_engine(engine_file_path):
    assert osp.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def bind_array_to_input(inp, bindings):
    import pycuda.driver as cuda
    input_buffer = np.ascontiguousarray(inp)
    input_memory = cuda.mem_alloc(inp.nbytes)
    bindings.append(int(input_memory))
    return input_buffer, input_memory


def bind_array_to_output(size, dtype, bindings):
    import pycuda.driver as cuda
    output_buffer = cuda.pagelocked_empty(size, dtype)
    output_memory = cuda.mem_alloc(output_buffer.nbytes)
    bindings.append(int(output_memory))
    return output_buffer, output_memory
