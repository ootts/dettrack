import math

import torch
import pycuda.driver as cuda

# import pycuda.autoinit
import tensorrt as trt
from torch.fx.experimental.fx2trt import torch_dtype_from_trt
from torchvision.ops import RoIAlign

from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.boxlist_ops import boxlist_iou, cat_boxlist
from disprcnn.utils.timer import EvalTime
from disprcnn.utils.trt_utils import load_engine, torch_device_from_trt
import torch.nn.functional as F


class YolactTrackingHeadInference:
    def __init__(self, engine_path):
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        self.logger = trt.Logger()
        engine = load_engine(engine_path)
        context = engine.create_execution_context()

        # prepare buffer
        cuda_inputs = {}
        cuda_outputs = {}
        bindings = []

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

        self.cuda_inputs = cuda_inputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

        self.roi_align = RoIAlign((7, 7), 69.0 / 550, 0)
        self.memory = torch.empty([0, 256, 7, 7], dtype=torch.float, device='cuda')  # todo: put in cfg
        self.boxmemory = None
        self.evaltime = EvalTime()

    def infer(self, x, ref_x):

        self.context.set_binding_shape(self.engine.get_binding_index("x"), (x.shape[0], 256, 7, 7))
        self.context.set_binding_shape(self.engine.get_binding_index("ref_x"), (ref_x.shape[0], 256, 7, 7))
        self.bindings = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(binding_idx))
            shape = tuple(self.context.get_binding_shape(binding_idx))
            device = torch_device_from_trt(self.engine.get_location(binding_idx))
            cuda_mem = torch.empty(size=shape, dtype=dtype, device=device)

            self.bindings.append(int(cuda_mem.data_ptr()))
            if self.engine.binding_is_input(binding):
                self.cuda_inputs[binding] = cuda_mem
            else:
                self.cuda_outputs[binding] = cuda_mem

        self.ctx.push()

        # restore
        stream = self.stream
        context = self.context
        engine = self.engine

        cuda_inputs = self.cuda_inputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        cuda_inputs['x'].copy_(x)
        cuda_inputs['ref_x'].copy_(ref_x)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        self.ctx.pop()

    def extract_roi_features(self, preds, feat):
        batchids = torch.cat([torch.full((len(boxlist), 1), i) for i, boxlist in enumerate(preds)]).cuda()
        roi_region = torch.cat([boxlist.bbox for boxlist in preds], dim=0)
        rois = torch.cat([batchids, roi_region], dim=1)
        roi_features = self.roi_align(feat, rois)
        return roi_features

    def track(self, left_preds, left_feat, right_preds, right_feat, width, height):

        self.evaltime("")
        confidence = left_preds[0].get_field('scores')
        roi_features = self.extract_roi_features(left_preds, left_feat)
        ref_x = self.memory
        x = roi_features
        self.evaltime("track:extract_roi_features")
        if len(self.memory) > 0 and x.numel() > 0:
            self.infer(x, ref_x)
            match_score = self.cuda_outputs['output']
        else:
            match_score = torch.empty([len(left_preds[0]), len(self.memory)], dtype=torch.float, device='cuda')
        self.evaltime("track:match score")
        ##############  ↓ Step : match  ↓  ##############
        if match_score.numel() > 0:
            match_score = F.softmax(match_score, dim=1)
            iou_score = self.calcIOUscore(left_preds)
            conf_score = torch.t(confidence.repeat(len(self.boxmemory) + 1, 1))
            match_score = torch.log(match_score) + 0.1 * torch.log(conf_score) + 10.0 * iou_score
            # TODO
            max_score, idxs = match_score.max(1)
            # todo fix bug for empty bin!!!
            # dual = match_score.max(0).indices[match_score.max(1).indices] == torch.arange(len(preds[0])).cuda()
            # matched = max_score > 0.5
            matched = (max_score > 0.0) & (idxs != 0)
            unmatched = ~matched
            matchidx = idxs[matched]
            duplicateid = []
            for id in matchidx:
                if id == 0:
                    continue
                if sum(matchidx == id) > 1:
                    if id not in duplicateid:
                        duplicateid.append(id)
            for id in duplicateid:
                conflict_box = (idxs == id) & matched
                idscores = match_score[..., id]
                idscores[~conflict_box] = -math.inf
                maxval, maxid = idscores.max(0)
                conflict_box[maxid] = False
                matched[conflict_box] = False

            matchidx = idxs[matched]

            # matched = matched & dual
        else:
            matched = torch.full([len(left_preds[0])], 0).cuda().bool()
            unmatched = ~matched
        cur_trackids = torch.full([len(left_preds[0])], -1).long().cuda()
        if matched.sum() > 0:
            idxs = idxs - 1
            cur_trackids[matched] = idxs[matched]
            cur_feat = roi_features[matched]
            self.memory[idxs[matched]] = cur_feat
            # todo:simplify
            self.boxmemory.bbox[idxs[matched]] = left_preds[0].bbox[matched]

        if unmatched.sum() > 0:
            new_tids = torch.arange(self.memory.shape[0], self.memory.shape[0] + unmatched.sum()).long().cuda()
            cur_trackids[unmatched] = new_tids
            self.memory = torch.cat([self.memory, roi_features[unmatched]], dim=0)
            # res_box = preds[0][unmatched]
            if not isinstance(self.boxmemory, BoxList):
                self.boxmemory = left_preds[0][unmatched]
            else:
                self.boxmemory = cat_boxlist([self.boxmemory, left_preds[0][unmatched]], ignore_fields=True,
                                             ignore_maps=True)
        keep = matched | unmatched

        left_pred = left_preds[0].resize([width, height])
        if left_pred.has_map('masks'):
            left_pred.add_map('masks', left_pred.get_map('masks').convert('mask').resize([width, height]))
        left_pred = left_pred[keep]
        cur_trackids = cur_trackids[keep]

        left_pred.add_field('trackids', cur_trackids)

        right_pred = right_preds[0].resize([width, height])
        self.evaltime("track:post process")
        return left_pred, right_pred

    def calcIOUscore(self, preds):
        if not isinstance(self.boxmemory, BoxList):
            return torch.empty([len(preds[0]), len(self.memory)], dtype=torch.float, device='cuda')
        ious = torch.zeros([len(preds[0]), 1], dtype=torch.float, device="cuda")
        ious = torch.cat([ious, boxlist_iou(preds[0], self.boxmemory)], dim=1)
        return ious

    def destory(self):
        self.ctx.pop()
        # self.logger.destroy()
        # del self.context
