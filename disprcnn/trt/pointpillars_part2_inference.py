import numpy as np
import torch
import pycuda.driver as cuda

# import pycuda.autoinit
import tensorrt as trt
from torch.fx.experimental.fx2trt import torch_dtype_from_trt

from disprcnn.modeling.models.psmnet.submodule import disparityregression
from disprcnn.utils.ppp_utils.box_coders import GroundBox3dCoderTorch
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

        self.box_coder = GroundBox3dCoderTorch()

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

    def predict(self, spatial_features, anchors, rect, Trv2c, P2, anchors_mask):
        self.infer(spatial_features)
        box_preds = self.cuda_outputs['box_preds'].clone()
        cls_preds = self.cuda_outputs['cls_preds'].clone()
        dir_cls_preds = self.cuda_outputs['dir_cls_preds'].clone()

        batch_size = anchors.shape[0]
        batch_anchors = anchors.view(batch_size, -1, 7)
        batch_rect = rect
        batch_Trv2c = Trv2c
        batch_P2 = P2
        batch_anchors_mask = anchors_mask.view(batch_size, -1)

        batch_box_preds = box_preds.view(batch_size, -1,
                                         self.box_coder.code_size)
        num_class_with_bg = 1

        batch_cls_preds = cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        batch_dir_preds = dir_cls_preds
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)

        from disprcnn.modeling.models.pointpillars import ops
        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_anchors_mask
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask.bool()]
                cls_preds = cls_preds[a_mask.bool()]
            if a_mask is not None:
                dir_preds = dir_preds[a_mask.bool()]
            dir_labels = torch.max(dir_preds, dim=-1)[1]
            # if self.encode_background_as_zeros:
            # this don't support softmax
            # assert self.use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)
            # else:
            #     encode background as first element in one-hot vector
            # if self.use_sigmoid_score:
            #     total_scores = torch.sigmoid(cls_preds)[..., 1:]
            # else:
            #     total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            # assert not self.use_rotate_nms
            # nms_func = box_torch_ops.rotate_nms
            # else:
            nms_func = ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)

            nms_score_threshold = 0.05
            thresh = torch.tensor(
                [nms_score_threshold],
                device=total_scores.device).type_as(total_scores)
            top_scores_keep = (top_scores >= thresh)
            top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if nms_score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    # if self.use_direction_classifier:
                    dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                box_preds_corners = ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = ops.corner_to_standup_nd(
                    box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=1000,  # self.nms_pre_max_size,
                    post_max_size=300,  # self.nms_post_max_size,
                    iou_threshold=0.5,  # self.nms_iou_threshold,
                )
            else:
                selected = None
            if selected is not None:
                selected_boxes = box_preds[selected]
                # if self.use_direction_classifier:
                selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                # if self.use_direction_classifier:
                dir_labels = selected_dir_labels
                # import pdb
                # pdb.set_trace()
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte().bool()
                box_preds[..., -1] += torch.where(
                    opp_labels.bool(), torch.tensor(np.pi).cuda().float(),
                    torch.tensor(0.0).cuda().float())
                # box_preds[..., -1] += (
                #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    # "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": torch.empty([0, 4], dtype=torch.float, device='cuda'),
                    "box3d_camera": torch.empty([0, 7], dtype=torch.float, device='cuda'),
                    "box3d_lidar": torch.empty([0, 7], dtype=torch.float, device='cuda'),
                    "scores": torch.empty([0, ], dtype=torch.float, device='cuda'),
                    "label_preds": torch.empty([0, ], dtype=torch.float, device='cuda'),
                    # "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)

    def destory(self):
        self.ctx.pop()
        del self.context
