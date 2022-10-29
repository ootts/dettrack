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
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs = {}
        cuda_inputs = {}
        host_outputs = []
        cuda_outputs = []
        bindings = []

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
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, left_images, right_images):
        self.ctx.push()
        N = left_images.shape[0]
        pad = np.zeros([20 - N, 3, 112, 112])
        left_images = np.concatenate([left_images, pad], axis=0)
        right_images = np.concatenate([right_images, pad], axis=0)
        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        np.copyto(host_inputs['left_input'], left_images.ravel())
        np.copyto(host_inputs['right_input'], right_images.ravel())
        for k in cuda_inputs.keys():
            cuda.memcpy_htod_async(cuda_inputs[k], host_inputs[k], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        self.ctx.pop()

    def destory(self):
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
    left_roi_images = left_roi_images.cpu().numpy()
    right_roi_images = right_roi_images.cpu().numpy()

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
