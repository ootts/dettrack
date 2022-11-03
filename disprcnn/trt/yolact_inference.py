import numpy as np
import cv2
import torch
import pycuda.driver as cuda

# import pycuda.autoinit
import tensorrt as trt
from disprcnn.utils.trt_utils import torch_dtype_from_trt

from disprcnn.modeling.models.yolact.layers.output_utils import postprocess
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.segmentation_mask import SegmentationMask
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import load_engine, torch_device_from_trt


class YolactInference:
    def __init__(self, engine_path, detector):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        # cuda_inputs = []
        # cuda_outputs = {}
        # bindings = []
        #
        # for binding in engine:
        #     binding_idx = engine.get_binding_index(binding)
        #     dtype = torch_dtype_from_trt(engine.get_binding_dtype(binding_idx))
        #     shape = tuple(engine.get_binding_shape(binding_idx))
        #     device = torch_device_from_trt(engine.get_location(binding_idx))
        #     cuda_mem = torch.empty(size=shape, dtype=dtype, device=device)
        #
        #     bindings.append(int(cuda_mem.data_ptr()))
        #     if engine.binding_is_input(binding):
        #         cuda_inputs.append(cuda_mem)
        #     else:
        #         cuda_outputs[binding] = cuda_mem
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        # self.cuda_inputs = cuda_inputs
        # self.cuda_outputs = cuda_outputs
        # self.bindings = bindings

        self.detector = detector
        self.evaltime = EvalTime()

    def infer(self, input_file1, input_file2):
        self.evaltime('')
        cuda_inputs = []
        cuda_outputs = {}
        bindings = []

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(binding_idx))
            shape = tuple(self.engine.get_binding_shape(binding_idx))
            device = torch_device_from_trt(self.engine.get_location(binding_idx))
            cuda_mem = torch.empty(size=shape, dtype=dtype, device=device)

            bindings.append(int(cuda_mem.data_ptr()))
            if self.engine.binding_is_input(binding):
                cuda_inputs.append(cuda_mem)
            else:
                cuda_outputs[binding] = cuda_mem

        self.ctx.push()
        # evaltime('yolact: prepare buffers')
        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        # host_inputs = self.host_inputs
        # cuda_inputs = self.cuda_inputs
        # host_outputs = self.host_outputs
        # cuda_outputs = self.cuda_outputs
        # bindings = self.bindings

        img1 = self.preprocess(input_file1)
        img2 = self.preprocess(input_file2)

        input_image = torch.stack([img1, img2]).cuda()
        cuda_inputs[0].copy_(input_image)
        self.evaltime('yolact: preprocess')
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        self.evaltime('yolact:execute')
        stream.synchronize()
        self.ctx.pop()

        return cuda_outputs

    def destory(self):
        self.ctx.pop()
        # del self.context

    def preprocess(self, input_file):
        image = cv2.imread(input_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.repeat(image[:, :, None], 3, axis=-1)
        image = image.astype(np.float32)
        img_h, img_w, _ = image.shape
        width, height = 600, 300
        image = cv2.resize(image, (width, height))
        image = image.astype(np.float32)
        mean = np.array([103.94, 116.78, 123.68])
        std = np.array([57.38, 57.12, 58.4])
        image = (image - mean) / std
        image = image[:, :, [2, 1, 0]]
        image = image.astype(np.float32)
        image = np.transpose(image, [2, 0, 1])
        image = torch.from_numpy(image).float()
        return image

    def postprocess(self, cuda_outputs):
        loc = cuda_outputs['loc']
        conf = cuda_outputs['conf']
        mask = cuda_outputs['mask']
        priors = cuda_outputs['priors']
        proto = cuda_outputs['proto']
        feat_out = cuda_outputs['feat_out']
        rets = self.detector({'loc': loc, 'conf': conf, 'mask': mask, 'priors': priors, 'proto': proto})
        return rets, feat_out

    def decode_preds(self, preds, h, w, add_mask):
        if preds[0]['detection'] is not None:
            preds = postprocess(preds, w, h, to_long=False)
            labels, scores, box2d, masks = preds
        else:
            box2d = torch.empty([0, 4], dtype=torch.float, device='cuda')
            labels = torch.empty([0], dtype=torch.long, device='cuda')
            scores = torch.empty([0], dtype=torch.float, device='cuda')
            masks = torch.empty([0, h, w], dtype=torch.long, device='cuda')
        boxlist = BoxList(box2d, (w, h))
        boxlist.add_field("labels", labels + 1)
        boxlist.add_field("scores", scores)
        if add_mask:
            keep = masks.sum(1).sum(1) > 20
            boxlist = boxlist[keep]
            masks = masks[keep]
            if add_mask:
                masks = SegmentationMask(masks, (w, h), mode='mask')
                boxlist.add_map("masks", masks)
        return [boxlist]

    def detect(self, input_file1, input_file2):

        cuda_outputs = self.infer(input_file1, input_file2)
        self.evaltime('')
        rets, feat_out = self.postprocess(cuda_outputs)
        self.evaltime('yolact:postprocess')
        left_rets, right_rets = [rets[0]], [rets[1]]
        left_feat, right_feat = feat_out[0:1], feat_out[1:]
        left_preds = self.decode_preds(left_rets, 300, 600, add_mask=True)
        right_preds = self.decode_preds(right_rets, 300, 600, add_mask=False)
        self.evaltime('yolact:decode')
        return left_preds, right_preds, left_feat, right_feat
