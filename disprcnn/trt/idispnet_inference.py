import torch
import pycuda.driver as cuda

# import pycuda.autoinit
import tensorrt as trt
from disprcnn.utils.trt_utils import torch_dtype_from_trt

from disprcnn.modeling.models.psmnet.submodule import disparityregression
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import load_engine, torch_device_from_trt
import torch.nn.functional as F


class IDispnetInference:
    def __init__(self, engine_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        # cuda_inputs = {}
        # cuda_outputs = {}
        # bindings = []

        # for binding in engine:
        #     binding_idx = engine.get_binding_index(binding)
        #     dtype = torch_dtype_from_trt(engine.get_binding_dtype(binding_idx))
        #     shape = tuple(engine.get_binding_shape(binding_idx))
        #     device = torch_device_from_trt(engine.get_location(binding_idx))
        #     cuda_mem = torch.empty(size=shape, dtype=dtype, device=device)
        #
        #     bindings.append(int(cuda_mem.data_ptr()))
        #     if engine.binding_is_input(binding):
        #         cuda_inputs[binding] = cuda_mem
        #     else:
        #         cuda_outputs[binding] = cuda_mem
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        # self.cuda_inputs = cuda_inputs
        # self.cuda_outputs = cuda_outputs
        # self.bindings = bindings
        self.evaltime = EvalTime()

    def infer(self, left_images, right_images):
        # evaltime = EvalTime('')
        # evaltime('')
        cuda_inputs = {}
        cuda_outputs = {}
        bindings = []
        self.context.set_binding_shape(self.engine.get_binding_index("left_input"),
                                       (left_images.shape[0], 3, 112, 112))
        self.context.set_binding_shape(self.engine.get_binding_index("right_input"),
                                       (right_images.shape[0], 3, 112, 112))
        # self.bindings = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(binding_idx))
            shape = tuple(self.context.get_binding_shape(binding_idx))
            device = torch_device_from_trt(self.engine.get_location(binding_idx))
            cuda_mem = torch.empty(size=shape, dtype=dtype, device=device)

            bindings.append(int(cuda_mem.data_ptr()))
            if self.engine.binding_is_input(binding):
                cuda_inputs[binding] = cuda_mem
            else:
                cuda_outputs[binding] = cuda_mem

        # evaltime('prep done')
        self.ctx.push()
        # evaltime("push ctx")
        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        # cuda_inputs = self.cuda_inputs
        # cuda_outputs = self.cuda_outputs
        # bindings = self.bindings

        cuda_inputs['left_input'].copy_(left_images)
        cuda_inputs['right_input'].copy_(right_images)
        # evaltime("idispnet:prep done")
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # evaltime("idispnet:execute_async_v2")
        stream.synchronize()
        self.ctx.pop()
        return cuda_outputs

    def destory(self):
        self.ctx.pop()
        # del self.context

    def predict_idisp(self, left, right):

        self.evaltime('')
        cuda_outputs = self.infer(left, right)
        self.evaltime('idispnet: infer')

        cost3 = cuda_outputs['output']
        cost3 = F.interpolate(cost3, [48, 112, 112], mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        p3 = disparityregression(pred3, 24, -24)
        self.evaltime('idispnet: disp reg')
        return p3
