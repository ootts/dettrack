# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

# transpose
from disprcnn.utils import cv2_util
from disprcnn.utils.pn_utils import to_array
from disprcnn.utils.timer import EvalTime

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimenion of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = list(image_size)  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}
        self.PixelWise_map = {}
        self.mask_thresh = 0.5

    def add_map(self, map, map_data):
        self.PixelWise_map[map] = map_data

    def get_map(self, map):
        return self.PixelWise_map[map]

    def has_map(self, map):
        return map in self.PixelWise_map

    def remove_map(self, map):
        self.PixelWise_map.pop(map)
        return self

    def maps(self):
        return list(self.PixelWise_map.keys())

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def _copy_map(self, bbox):
        for k, v in bbox.PixelWise_map.items():
            self.PixelWise_map[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        bbox._copy_map(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box
        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            bbox._copy_map(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, (torch.Tensor, int)):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def rotate(self, angle, *args, **kwargs):
        bbox = BoxList(self.bbox, self.size, self.mode)
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.rotate(angle, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox

    def scale(self, scale, *args, **kwargs):
        bbox = BoxList(self.bbox, self.size, self.mode)
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.scale(scale, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        for k, v in self.PixelWise_map.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_map(k, v)
        return bbox

    def __getitem__(self, item):
        if isinstance(item, list) and isinstance(item[0], torch.Tensor):
            item = [i.item() for i in item]  # compat to onnx
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        # bbox._copy_map(self)
        for k, v in self.extra_fields.items():
            if isinstance(v, int):
                bbox.add_field(k, v)
            else:
                bbox.add_field(k, v[item])
        for k, v in self.PixelWise_map.items():
            if hasattr(v, '__getitem__'):
                bbox.add_map(k, v[item])
            else:
                bbox.add_map(k, v)
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def remove_small_area(self, thresh=100):
        masks = self.extra_fields['masks']
        keep = []
        max_area = 0
        for i, polygon in enumerate(masks):
            mask = np.ascontiguousarray(polygon.convert(mode='mask').numpy().astype(np.uint8)[:, :, None])
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(contour) for contour in contours]
            if max(areas) > max_area:
                max_keep = i
            if max(areas) >= thresh:
                keep.append(i)
        if len(keep) == 0:
            keep = [max_keep]
        return self[keep]

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")
        return area

    def copy_with_fields(self, fields, maps=None):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            bbox.add_field(field, self.get_field(field))
        if maps is not None:
            if not isinstance(maps, (list, tuple)):
                maps = [maps]
            for map in maps:
                bbox.add_map(map, self.get_map(map))
        return bbox

    def convert_to_kiiti_label(self, frame_id, P=None, calib_filepath=None, seq_id=None, score_threshold=-1000):
        """
        Convert result of current frame to the form of KITTI label, which is:

        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    frame        Frame within the sequence where the object appearers
           1    track id     Unique tracking id of this object within this sequence
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Integer (0,1,2) indicating the level of truncation.
                             Note that this is in contrast to the object detection
                             benchmark where truncation is a float in [0,1].
           1    occluded     Integer (0,1,2,3) indicating occlusion state::
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.

        :return:
        """

        def box_2d_from_corners(inputs):
            """
            Obtain 2D bounding box corners from 3D bounding box corners
            :param box_corners: [N, 8, 2], 2D coordinate for the 8 corners of the 3D bounding box after projection
            :return: [N, 4], 2D coordinates for 2D bounding box
            """
            box_corners, valid = inputs
            box2d = []
            bs = box_corners.shape[0]
            for b in range(bs):
                bbox_2d_batch = box_corners[b]
                valid_batch = valid[b]
                if valid_batch.any() == False:
                    box2d.append(torch.ones((0, 4)))
                    continue
                bbox_2d_batch = bbox_2d_batch[valid_batch]

                box_3d_2d_min = torch.min(bbox_2d_batch, dim=0)[0]
                box_3d_2d_max = torch.max(bbox_2d_batch, dim=0)[0]

                min_xy = torch.Tensor([0, 0])
                box_3d_2d_min = torch.max(box_3d_2d_min, min_xy)
                max_xy = torch.Tensor(self.size) - 1
                box_3d_2d_max = torch.min(box_3d_2d_max, max_xy)

                box_3d_2d = torch.cat((box_3d_2d_min, box_3d_2d_max), dim=0)
                box2d.append(box_3d_2d)

            box2d = torch.stack(box2d)

            # upper_left = torch.min(box_corners, 1)[0]
            # lower_right = torch.max(box_corners, 1)[0]
            # min_xy = torch.Tensor([0, 0]).unsqueeze(0)
            # upper_left = torch.max(upper_left, min_xy)
            # max_xy = torch.Tensor(self.size).unsqueeze(0) - 1
            # lower_right = torch.min(lower_right, max_xy)
            # box2d = torch.cat((upper_left, lower_right), dim=1)
            return box2d.numpy()

        def get_alpha(x, z, ry):
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry
            return alpha

        def result2str(track_id_list, score_list, box2d_list, box3d_list, score_threshod):
            """

            :param track_id_list: [N], np.ndarray
            :param score_list: [N], np.ndarray
            :param box2d_list: [N, 4] np.ndarray
            :param box3d_list: [N, 7], np.ndarray, ry_lhwxyz
            :return:
            """
            # remove results with duplicate id
            if len(np.unique(track_id_list)) != len(track_id_list):
                # print(track_id_list)
                # print(np.unique(track_id_list))
                # print(score_list)
                u, c = np.unique(track_id_list, return_counts=True)
                dup_list = u[c > 1]
                for dup in dup_list:
                    # print(dup)
                    dup_idx = np.where(track_id_list == dup)[0]
                    # print(dup_idx)
                    dup_scores = score_list[dup_idx]
                    # print(dup_scores)
                    if np.sum(dup_scores) / len(dup_scores) - dup_scores[0] <= 0.5:
                        remove_idx = dup_idx[1:]
                    else:
                        remove_idx = dup_idx[np.where(dup_scores != np.max(dup_scores))[0]]
                    track_id_list = np.delete(track_id_list, remove_idx)
                    score_list = np.delete(score_list, remove_idx)
                    box2d_list = np.delete(box2d_list, remove_idx, 0)
                    box3d_list = np.delete(box3d_list, remove_idx, 0)

            result_str = ""
            keep_id_list = []
            for idx, (track_id, score, box2d, box3d) \
                    in enumerate(zip(track_id_list, score_list, box2d_list, box3d_list)):
                if score >= score_threshold:
                    result_str += "{} {} Car 0 0 {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                        frame_id, track_id,
                        get_alpha(box3d[-3], box3d[-1], box3d[0]),
                        box2d[0], box2d[1], box2d[2], box2d[3],
                        box3d[2], box3d[3], box3d[1],
                        box3d[4], box3d[5], box3d[6],
                        box3d[0], score)
                    keep_id_list.append(idx)

            return result_str, keep_id_list

        # Main program
        if not (P is not None or calib_filepath is not None or self.has_field('P')):
            P = torch.ones([3, 4])
            if seq_id is not None:
                print("[ERROR] {} {}!!!".format(seq_id, frame_id))

        # assert P is not None or calib_filepath is not None or self.has_field('P')

        if calib_filepath is not None:
            import os
            from maskrcnn_benchmark.utils.kitti_utils import Calibration
            calib = Calibration(calib_filepath, from_video=True)
            P = calib.P  # [3, 4]
        elif self.has_field('P'):
            P = self.get_field('P')  # [3, 4]
            P = torch.tensor(P, dtype=torch.float32)

        # frontend values
        det_id = self.get_field('det_id').numpy()
        pred_score = self.get_field('box3d_score').numpy()
        pred_box2d = box_2d_from_corners(self.get_field('box3d').project_to_2d(P))
        pred_box3d = self.get_field('box3d').convert('ry_lhwxyz').bbox_3d.numpy()

        backend_id = self.get_field('box3d_backend_ids').numpy()
        backend_score = self.get_field('score_backend').numpy()
        backend_box2d = box_2d_from_corners(self.get_field('box3d_backend').project_to_2d(P))
        backend_box3d = self.get_field('box3d_backend').convert('ry_lhwxyz').bbox_3d.numpy()

        pred_result_str = ''
        backend_result_str, keep_id_list = \
            result2str(backend_id, backend_score, backend_box2d, backend_box3d, score_threshold)

        if len(backend_id) != len(np.unique(backend_id)):
            a = 1 + 1

        if len(keep_id_list) > 0:
            track_id_list = np.array(backend_id)[np.array(keep_id_list)]
        else:
            track_id_list = []
        if len(track_id_list) != len(np.unique(track_id_list)):
            a = 1 + 1

        return pred_result_str, backend_result_str, track_id_list

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

    def plot(self, img=None, show=False, calib=None, draw_mask=True, ignore_2d_when_3d_exists=False, class_names=None,
             **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        plt.close("all")
        plt.axis('off')
        plt.tight_layout(pad=0)
        if img is not None:
            plt.imshow(to_array(img))
        colors = list(mcolors.BASE_COLORS.keys())
        if self.has_field("trackids"):
            trackids = self.get_field("trackids").tolist()
        else:
            trackids = list(range(len(self)))
        for i, box in enumerate(self.convert('xywh').bbox.tolist()):
            x, y, w, h = box
            c = colors[trackids[i] % len(colors)]
            if not (ignore_2d_when_3d_exists and self.has_field("box3d")):
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, color=c))
            text = "tid " + str(trackids[i])
            label = self.get_field("labels").tolist()[i]
            if class_names is None:
                text = text + f" label {label}"
            else:
                text = text + " " + class_names[label - 1]

            if self.has_field('scores'):
                text = text + " " + '%.2f' % self.get_field('scores').tolist()[i]
                plt.text(x, y, text, color=c, **kwargs)
        if draw_mask:
            if self.has_field('mask'):
                masks = self.get_field('mask')
                from disprcnn.modeling.roi_heads.mask_head.inference import Masker
                masks = Masker()([masks], [self])[0].squeeze(1).cpu().byte().numpy()
                for m in masks:
                    contour, hierarchy = cv2_util.findContours(
                        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
                    )
                    for c in contour:
                        c = c.squeeze(1)
                        plt.gca().add_patch(plt.Polygon(c, fill=False, **kwargs))
            elif self.has_field('masks'):
                masks = self.get_field('masks').cpu().byte().numpy()
                for i, m in enumerate(masks):
                    contour, hierarchy = cv2_util.findContours(
                        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
                    )
                    for c in contour:
                        c = c.squeeze(1)
                        plt.gca().add_patch(plt.Polygon(c, fill=False, color=colors[i % len(colors)]))
            elif self.has_map("masks"):
                masks = self.get_map('masks').convert('mask').get_mask_tensor(squeeze=False).cpu().numpy().astype(
                    np.uint8)
                for i, m in enumerate(masks):
                    contour, hierarchy = cv2_util.findContours(
                        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
                    )
                    for c in contour:
                        c = c.squeeze(1)
                        plt.gca().add_patch(plt.Polygon(c, fill=False, color=colors[trackids[i] % len(colors)]))
        if self.has_field("box3d") and calib is not None:
            predcorners = self.get_field('box3d').convert('corners').bbox_3d.view(-1, 8, 3)
            corners = calib.corners3d_to_img_boxes(predcorners)[1].cpu().numpy()
            for ci, c in enumerate(corners):
                color = colors[trackids[ci] % len(colors)]
                pts = np.array([c[[0, 1, 2, 3, 0, 4, 7, 3]]])
                from matplotlib import collections as mc
                seg = mc.LineCollection(pts, colors=color, linewidths=1, )
                plt.gca().add_collection(seg)
                pts = np.array([c[[5, 4, 7, 6, 5, 1, 2, 6]]])
                seg = mc.LineCollection(pts, colors=color, linewidths=1)
                plt.gca().add_collection(seg)
        if show:
            plt.show()

    def compress(self):
        if self.has_field("masks"):
            import pycocotools.mask as mask_utils
            et = EvalTime()
            et('')
            bitmasks = self.extra_fields['masks']
            tmp = np.ascontiguousarray(bitmasks.byte().cpu().numpy().transpose(1, 2, 0))
            et('tmp')
            rlemasks = mask_utils.encode(np.asfortranarray(tmp))
            et('encode')
            # masks = mask_utils.encode(np.asfortranarray(masks))
            self.add_field("masks", rlemasks)
        elif self.has_map("masks"):
            raise NotImplementedError()
            print()
        return self

    @property
    def widths(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            widths = box[:, 2] - box[:, 0] + TO_REMOVE
        elif self.mode == 'xywh':
            widths = bbox[:, 2]
        else:
            raise RuntimeError("Should not be here")
        return widths

    @property
    def heights(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 1
            heights = box[:, 3] - box[:, 1] + TO_REMOVE
        elif self.mode == 'xywh':
            heights = bbox[:, 3]
        else:
            raise RuntimeError("Should not be here")
        return heights

    @property
    def height(self):
        return self.size[1]

    @property
    def width(self):
        return self.size[0]

    @property
    def device(self):
        return self.bbox.device


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
