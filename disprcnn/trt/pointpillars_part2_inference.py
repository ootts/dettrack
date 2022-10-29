import torch
import pycuda.driver as cuda

# import pycuda.autoinit
import tensorrt as trt
from torch.fx.experimental.fx2trt import torch_dtype_from_trt

from disprcnn.modeling.models.psmnet.submodule import disparityregression
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import load_engine, torch_device_from_trt
import torch.nn.functional as F


class PointPillarsPart2Inference:
    def __init__(self, engine_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        cuda_inputs = {}
        cuda_outputs = {}
        bindings = []

        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            dtype = torch_dtype_from_trt(engine.get_binding_dtype(binding_idx))
            shape = tuple(engine.get_binding_shape(binding_idx))
            device = torch_device_from_trt(engine.get_location(binding_idx))
            cuda_mem = torch.empty(size=shape, dtype=dtype, device=device)

            bindings.append(int(cuda_mem.data_ptr()))
            if engine.binding_is_input(binding):
                cuda_inputs[binding] = cuda_mem
            else:
                cuda_outputs[binding] = cuda_mem
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.cuda_inputs = cuda_inputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, spatial_features):
        evaltime = EvalTime()
        self.ctx.push()

        evaltime("")
        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        cuda_inputs = self.cuda_inputs
        bindings = self.bindings

        cuda_inputs['input'].copy_(spatial_features)
        evaltime("prep done")
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        evaltime("pointpillars part2 infer")
        stream.synchronize()
        self.ctx.pop()

    def destory(self):
        self.ctx.pop()
        del self.context
