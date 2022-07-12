import torch

# from disprcnn.modeling.roi_heads.mask_head.inference import Masker
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.segmentation_mask import SegmentationMask, BinaryMaskList
from disprcnn.utils.stereo_utils import expand_box_to_integer


def sanity_check(left_predictions, right_predictions):
    assert len(left_predictions) == len(right_predictions)
    assert all(isinstance(l, BoxList) for l in left_predictions)
    assert all(isinstance(r, BoxList) for r in right_predictions)
    assert all(len(l) == len(r) for l, r in zip(left_predictions, right_predictions))
    assert all(l.size == r.size for l, r in zip(left_predictions, right_predictions))


class DisparityMapProcessor:
    def _forward_single_image(self, left_prediction: BoxList, right_prediction: BoxList) -> DisparityMap:
        left_bbox = left_prediction.bbox
        right_bbox = right_prediction.bbox
        disparity_or_depth_preds = left_prediction.get_field('disparity')
        mask_pred = left_prediction.get_map('masks').get_mask_tensor(squeeze=False)
        num_rois = len(left_bbox)
        disparity_maps_per_img = []
        if num_rois != 0:
            for left_roi, right_roi, disp_or_depth_roi, mask in zip(left_bbox, right_bbox,
                                                                    disparity_or_depth_preds, mask_pred):
                x1, y1, x2, y2 = expand_box_to_integer(left_roi.tolist())
                x1p, _, x2p, _ = expand_box_to_integer(right_roi.tolist())
                depth_map_per_roi = torch.zeros((left_prediction.height, left_prediction.width)).cuda()
                disparity_map_per_roi = torch.zeros_like(depth_map_per_roi)
                disp_roi = DisparityMap(disp_or_depth_roi).resize(
                    (max(x2 - x1, x2p - x1p), y2 - y1),
                    mode='inverse_bilinear',
                    x1_minus_x1p=x1 - x1p).crop(
                    (0, 0, x2 - x1, y2 - y1)).data
                disp_roi = disp_roi + x1 - x1p
                disparity_map_per_roi[y1:y2, x1:x2] = disp_roi
                disparity_map_per_roi = disparity_map_per_roi * mask.float().cuda()
                disparity_maps_per_img.append(disparity_map_per_roi)
        if len(disparity_maps_per_img) != 0:
            disparity_maps_per_img = torch.stack(disparity_maps_per_img).sum(dim=0)
        else:
            disparity_maps_per_img = torch.zeros((left_prediction.height, left_prediction.width)).cuda()
        return DisparityMap(disparity_maps_per_img)

    def __call__(self, left_predictions, right_predictions):
        if isinstance(left_predictions, BoxList) and isinstance(right_predictions, BoxList):
            left_predictions = [left_predictions]
            right_predictions = [right_predictions]
        sanity_check(left_predictions, right_predictions)
        results = []
        for l, r in zip(left_predictions, right_predictions):
            results.append(self._forward_single_image(l, r))
        if len(results) == 1:
            results = results[0]
        return results


def clip_mask_to_minmaxdisp(mask, dispairty, leftbox, rightbox, mindisp=-48, maxdisp=48, resolution=28):
    mask = mask.clone()
    disparity_map = DisparityMap(dispairty)
    for lb, rb in zip(leftbox, rightbox):
        x1, y1, x2, y2 = lb.tolist()
        x1p, _, x2p, _ = rb.tolist()
        max_width = max(x2 - x1, x2p - x1p)
        roi_disparity = disparity_map.crop(lb.tolist()).data
        roi_disparity = roi_disparity - (x1 - x1p)
        roi_mask = mask[round(y1):round(y2), round(x1):round(x2)]
        roi_mask = roi_mask & (roi_disparity * resolution * 4 / max_width > mindisp).byte() & (
                roi_disparity * resolution * 4 / max_width < maxdisp).byte()
        # roi_mask[roi_disparity * resolution * 4 / (x2 - x1) < mindisp] = 0
        # roi_mask[roi_disparity * resolution * 4 / (x2 - x1) > maxdisp] = 0
        # mask[round(y1):round(y2), round(x1):round(x2)] = roi_mask
        mask[round(y1):round(y2), round(x1):round(x2)] = mask[round(y1):round(y2), round(x1):round(x2)] & roi_mask
    return mask


# def clip_mask_to_minmaxdisp_wo_resize(mask, disparity, leftbox, rightbox, mindisp=-48, maxdisp=144):
#     mask = mask.clone()
#     disparity_map = DisparityMap(disparity)
#     for lb, rb in zip(leftbox, rightbox):
#         x1, y1, x2, y2 = lb.tolist()
#         x1p, _, x2p, _ = rb.tolist()
#         roi_disparity = disparity_map.crop(lb.tolist()).data
#         roi_disparity = roi_disparity - (x1 - x1p)
#         roi_mask = mask[round(y1):round(y2), round(x1):round(x2)]
#         roi_mask[roi_disparity < mindisp] = 0
#         roi_mask[roi_disparity > maxdisp] = 0
#         mask[round(y1):round(y2), round(x1):round(x2)] = roi_mask
#     return mask


def post_process_and_resize_prediction(left_prediction: BoxList, right_prediction: BoxList, dst_size=(1280, 720),
                                       threshold=0.7,
                                       padding=1, mask_aware_resize=False, process_disparity=True):
    if mask_aware_resize:
        raise NotImplementedError()
    left_prediction = left_prediction.clone()
    right_prediction = right_prediction.clone()
    if process_disparity and not left_prediction.has_map('disparity'):
        disparity_map_processor = DisparityMapProcessor()
        disparity_pred_full_img = disparity_map_processor(
            left_prediction, right_prediction)
        left_prediction.add_map('disparity', disparity_pred_full_img)
    left_prediction = left_prediction.resize(dst_size)
    right_prediction = right_prediction.resize(dst_size)
    mask_pred = left_prediction.get_field('mask')
    masker = Masker(threshold=threshold, padding=padding)
    mask_pred = masker([mask_pred], [left_prediction])[0].squeeze(1)
    if mask_pred.shape[0] != 0:
        # mask_preds_per_img = mask_pred.sum(dim=0)[0].clamp(max=1)
        mask_preds_per_img = mask_pred
    else:
        mask_preds_per_img = torch.zeros((1, *dst_size[::-1]))
    left_prediction.add_field('mask', mask_preds_per_img)
    return left_prediction, right_prediction

# def get_disp_gt_and_mask_gt(ds):
#     disparity_gt_tensors, mask_gts = [], []
#     cache_path = '.cache/disp_targets.pkl'
#     if os.path.exists(cache_path):
#         disparity_gt_tensors = pickle.load(open(cache_path, 'rb'))
#     else:
#         for idx in range(len(ds)):
#             idxanno = ds.get_targets(idx)[0]
#             leftanno, rightanno = idxanno['left'], idxanno['right']
#             disparity_gt = leftanno.get_map('disparity')
#             disparity_gt_tensor = disparity_gt.data
#             disparity_gt_tensors.append(disparity_gt_tensor)
#         pickle.dump(disparity_gt_tensors, open(cache_path, 'wb'))
#     cache_path = '.cache/mask_gts.pkl'
#     if os.path.exists(cache_path):
#         mask_gts = pickle.load(open(cache_path, 'rb'))
#     else:
#         for idx in range(len(ds)):
#             idxanno = ds.get_targets(idx)[0]
#             leftanno, rightanno = idxanno['left'], idxanno['right']
#             mask_gt = leftanno.get_field('masks').get_full_image_mask_tensor()
#             mask_gts.append(mask_gt)
#         pickle.dump(mask_gts, open(cache_path, 'wb'))
#     return disparity_gt_tensors, mask_gts
#
#
# def evaluation(ds: GTAKITTIDatasetCocoFormat,
#                left_predictions,
#                right_predictions,
#                erose_size=5, erose_iteration=0):
#     ds = deepcopy(ds)
#     ds.transforms = None
#     am = AverageMeter()
#     # left_predictions, right_predictions = predictions['left'], predictions['right']
#     epes = []
#     eval_masks = []
#     # ignores = [22, 45, 76, 173, 174, 175, *range(339, 370), 564, ]
#     # bigcars = [15, 16, 17, 18, 21, 22, 44, 45, 46, 71, 72, 73, 74, 75, 78, 171, 145, 311, 564
#     #     , 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584]
#     disparity_gt_tensors, mask_gts = get_disp_gt_and_mask_gt(ds)
#     for idx in progressbar(range(len(ds))):
#         # if idx in ignores:
#         #     eval_masks.append(None)
#         #     continue
#         mask_gt = mask_gts[idx]
#         disparity_gt_tensor = disparity_gt_tensors[idx]
#         # get i-th predictionx
#         idx_left_prediction = left_predictions[idx]
#         idx_right_prediction = right_predictions[idx]
#         # post process prediction
#         idx_left_prediction, idx_right_prediction = post_process_and_resize_prediction(
#             idx_left_prediction, idx_right_prediction)
#         # retrieve predictions' elements
#         disp_pred = idx_left_prediction.get_map('disparity').data
#         mask_pred_full_img = idx_left_prediction.get_field('mask')
#         mask_gt_pred = (mask_pred_full_img.int()) & (mask_gt.int())
#         mask_pred_gt_in_minmaxdisp = clip_mask_to_minmaxdisp(mask_gt_pred,
#                                                              disparity_gt_tensor,
#                                                              idx_left_prediction.bbox,
#                                                              idx_right_prediction.bbox,
#                                                              mindisp=-48, maxdisp=48)
#         mask_pred_gt_in_minmaxdisp = mask_pred_gt_in_minmaxdisp & ((disparity_gt_tensor < 192).int())
#         mask_pred_gt_in_minmaxdisp_erose = torch.Tensor(cv2.erode(
#             mask_pred_gt_in_minmaxdisp.byte().numpy(),
#             np.ones((erose_size, erose_size), np.uint8),
#             iterations=erose_iteration))
#         eval_masks.append(mask_pred_gt_in_minmaxdisp_erose)
#         epe = end_point_error(disparity_gt_tensor, mask_pred_gt_in_minmaxdisp_erose, disp_pred)
#         am.update(epe, mask_pred_gt_in_minmaxdisp.sum().item())
#         epes.append(epe)
#     logger = logging.getLogger("disprcnn.inference")
#     logger.info('end-point-error ' + str(am.avg))
#     torch.save(eval_masks, 'tmp/eval_masks.pth')
#     return epes, eval_masks
#
#
# def compare_epe(eval_masks, disprcnn_epes):
#     cache_path = '.cache/gta_full_image_preds_1280x720.pth'
#     if os.path.exists(cache_path):
#         big_preds = torch.load(cache_path)
#     else:
#         predictions = torch.load('tmp/gta_full_image_preds.pth', 'cpu')
#         big_preds = [DisparityMap(pred).resize((1280, 720)).data for pred in predictions]
#         torch.save(big_preds, cache_path)
#     targets = pickle.load(open('.cache/disp_targets.pkl', 'rb'))
#     print('Computing epe with original size 1280x720 and eval_masks.')
#     am = AverageMeter()
#     psmnet_epes = []
#     final_masks = []
#     for i, (pred, target, mask) in enumerate(
#             progressbar(zip(big_preds, targets, eval_masks), max_value=len(big_preds))):
#         epe = end_point_error(target, mask, pred)
#         am.update(epe, mask.sum().item())
#         psmnet_epes.append(epe)
#         final_masks.append(mask.sum().item())
#     print('Average epe', am.avg)
#     mask_cnt = [m.sum().item() for m in eval_masks]
#
#     d = torch.tensor(disprcnn_epes)
#     p = torch.tensor(psmnet_epes)
#
#     print('better', 'disprcnn<=psmnet', (d <= p).sum().item())
#
#     worse = (d > p).nonzero().squeeze()
#     better = (d <= p).nonzero().squeeze()
#     slight_better = (d <= p + 1.0).nonzero().squeeze()
#
#     bigcars = [15, 16, 17, 18, 21, 22, 44, 45, 46, 71, 72, 73, 74, 75, 78, 171, 145, 311, 564
#         , 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584]
#     valds_minus_bigcars = list(set(list(range(612))) - set(bigcars))
#
#     better_p = ((p * torch.tensor(mask_cnt).float())[better].sum().item() / (
#         torch.tensor(mask_cnt)[better].sum())).item()
#     better_d = ((d * torch.tensor(mask_cnt).float())[better].sum().item() / (
#         torch.tensor(mask_cnt)[better].sum())).item()
#     print('better', better_d, 'vs', better_p)
#
#     small_car_p = ((p * torch.tensor(mask_cnt).float())[valds_minus_bigcars].sum().item() / (
#         torch.tensor(mask_cnt)[valds_minus_bigcars].sum())).item()
#     small_car_d = ((d * torch.tensor(mask_cnt).float())[valds_minus_bigcars].sum().item() / (
#         torch.tensor(mask_cnt)[valds_minus_bigcars].sum())).item()
#     print('remove big car', small_car_d, 'vs', small_car_p)
