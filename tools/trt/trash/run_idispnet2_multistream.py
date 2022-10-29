import os.path as osp
import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from disprcnn.utils.trt_utils import load_engine


class Inference:
    def __init__(self, engine_path, max_batch_size=6):
        self.ctx = cuda.Device(0).make_context()
        streams = []
        for i in range(max_batch_size):
            stream = cuda.Stream()
            streams.append(stream)
        TRT_LOGGER = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        all_host_inputs = []
        all_cuda_inputs = []
        all_host_outputs = []
        all_cuda_outputs = []
        all_bindings = []
        for i in range(max_batch_size):
            bindings = []
            host_inputs = {}
            cuda_inputs = {}
            host_outputs = []
            cuda_outputs = []
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding))
                host_mem = cuda.pagelocked_empty(size, np.float32)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)

                bindings.append(int(cuda_mem))
                if engine.binding_is_input(binding):
                    host_inputs[binding] = host_mem
                    cuda_inputs[binding] = cuda_mem
                else:
                    host_outputs.append(host_mem)
                    cuda_outputs.append(cuda_mem)
            all_host_inputs.append(host_inputs)
            all_cuda_inputs.append(cuda_inputs)
            all_host_outputs.append(host_outputs)
            all_cuda_outputs.append(cuda_outputs)
            all_bindings.append(bindings)
        # store
        self.streams = streams
        self.context = context
        self.engine = engine

        self.all_host_inputs = all_host_inputs
        self.all_cuda_inputs = all_cuda_inputs
        self.all_host_outputs = all_host_outputs
        self.all_cuda_outputs = all_cuda_outputs
        self.all_bindings = all_bindings

    def infer(self, left_images, right_images):
        self.ctx.push()
        N = left_images.shape[0]
        # restore
        streams = self.streams
        context = self.context
        engine = self.engine

        all_host_inputs = self.all_host_inputs
        all_cuda_inputs = self.all_cuda_inputs
        all_host_outputs = self.all_host_outputs
        all_cuda_outputs = self.all_cuda_outputs
        all_bindings = self.all_bindings
        for i in range(N):
            np.copyto(all_host_inputs[i]['left_input'], left_images[i][None].ravel())
            np.copyto(all_host_inputs[i]['right_input'], right_images[i][None].ravel())
            for k in all_cuda_inputs[i].keys():
                cuda.memcpy_htod_async(all_cuda_inputs[i][k], all_host_inputs[i][k], streams[i])
            context.execute_async_v2(bindings=all_bindings[i], stream_handle=streams[i].handle)
            cuda.memcpy_dtoh_async(all_host_outputs[i][0], all_cuda_outputs[i][0], streams[i])
        for i in range(N):
            streams[i].synchronize()
        self.ctx.pop()

    def destory(self):
        # pass
        self.ctx.pop()


def main():
    from disprcnn.engine.defaults import setup
    from disprcnn.engine.defaults import default_argument_parser
    from disprcnn.data import make_data_loader
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/drcnn/kitti_tracking/pointpillars_112_600x300_demo.yaml'
    cfg = setup(args)

    left_roi_images, right_roi_images = torch.load('tmp/left_right_roi_images.pth')
    left_roi_images = left_roi_images.cpu().numpy()[0:2]
    right_roi_images = right_roi_images.cpu().numpy()[0:2]

    engine_file = osp.join(cfg.trt.convert_to_trt.output_path, "idispnet.engine")
    inferencer = Inference(engine_file)
    inferencer.infer(left_roi_images, right_roi_images)
    inferencer.destory()

    print()

    ref = torch.load('tmp/outputs.pth')
    print()
    # with load_engine(engine_file) as engine:
    #     for _ in range(10000):
    # infer(engine, left_roi_images, right_roi_images)


if __name__ == '__main__':
    main()
