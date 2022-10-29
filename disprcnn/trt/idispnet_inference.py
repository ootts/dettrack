import torch
import pycuda.driver as cuda

# import pycuda.autoinit
import tensorrt as trt
from torch.fx.experimental.fx2trt import torch_dtype_from_trt

from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import load_engine, torch_device_from_trt


class IDispnetInference:
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

    def infer(self, left_images, right_images):
        evaltime = EvalTime()
        self.ctx.push()

        evaltime("")
        N = left_images.shape[0]
        left_pad = torch.zeros([20 - N, 3, 112, 112]).float().to(left_images.device)
        left_images = torch.cat([left_images, left_pad], dim=0)
        right_pad = torch.zeros([20 - N, 3, 112, 112]).float().to(left_images.device)
        right_images = torch.cat([right_images, right_pad], dim=0)
        evaltime("pad")

        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        cuda_inputs = self.cuda_inputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # img1 = self.preprocess(input_file1)
        # img2 = self.preprocess(input_file2)

        # input_image = torch.stack([img1, img2]).cuda()
        # np.copyto(host_inputs[0], input_image.ravel())
        cuda_inputs['left_input'].copy_(left_images)
        cuda_inputs['right_input'].copy_(right_images)
        evaltime("prep done")
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        evaltime("idispnet infer")
        stream.synchronize()
        self.ctx.pop()

    def destory(self):
        self.ctx.pop()
        del self.context
