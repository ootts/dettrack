import enum
import math

import loguru
import numpy as np
import torch
import tqdm
from dl_ext.primitive import safe_zip
from pytorch3d.transforms import se3_log_map

# from disprcnn.metric.accuracy import accuracy
from disprcnn.registry import EVALUATORS
from disprcnn.utils import comm
import os


@EVALUATORS.register('star_pose_eval')
def build(cfg):
    def f(x, trainer):
        if comm.get_rank() == 0 and 'obj_pose' in x[0]:
            pred_obj_poses = torch.stack([a['obj_pose'] for a in x])
            gt_obj_poses = torch.cat([x['object_pose'] for x in trainer.valid_dl])
            pred_dof6 = se3_log_map(pred_obj_poses.permute(0, 2, 1))
            gt_dof6 = se3_log_map(gt_obj_poses.permute(0, 2, 1))
            init_dof6 = torch.tensor(trainer.cfg.model.star.pose_init)
            init_err = (init_dof6 - gt_dof6).abs()
            pred_err = (pred_dof6 - gt_dof6).abs()
            loguru.logger.info(f"init dof6 err,{init_err.mean().item():.4f}")
            loguru.logger.info(f"init trans err,{init_err[..., :3].mean().item():.4f}")
            loguru.logger.info(f"init trans err max,{init_err[..., :3].max().item():.4f}")
            loguru.logger.info(f"init rot err,{math.degrees(init_err[..., 3:].mean().item()):.4f}")
            loguru.logger.info(f"init rot err max,{math.degrees(init_err[..., 3:].max().item()):.4f}")

            loguru.logger.info(f"optimized dof6 err, {pred_err.mean().item():.4f}")
            loguru.logger.info(f"optimized trans err, {pred_err[..., :3].mean().item():.4f}")
            loguru.logger.info(f"optimized trans err max, {pred_err[..., :3].mean(-1).max().item():.4f}")
            loguru.logger.info(f"optimized rot err, {math.degrees(pred_err[..., 3:].mean().item()):.4f}")
            loguru.logger.info(f"optimized rot err max, {math.degrees(pred_err[..., 3:].max().item()):.4f}")

    return f


@EVALUATORS.register('print_loss')
def build(cfg):
    def f(x, ds):
        for i, d in enumerate(x):
            print('-' * 10, i, '-' * 10)
            for k, v in d.get('metrics', {}).items():
                print(k, v.item())
            for k, v in d.get('loss_dict', {}).items():
                print(k, v.item())

    return f


# def build_evaluators(cfg):
#     evaluators = []
#     for e in cfg.test.evaluators:
#         evaluators.append(EVALUATORS[e](cfg))
#     return evaluators
@EVALUATORS.register("kittiobj")
def build(cfg):
    def kittiobjeval(predictions, trainer):
        PROJECT_ROOT = "/home/songyunzhou/dettrack"
        label = "car"
        output_folder = os.path.join(PROJECT_ROOT, cfg.output_dir,"evaluate", 'txt')
        os.makedirs(output_folder, exist_ok=True)
        for i, pred in enumerate(tqdm.tqdm(predictions)):
            limgid = pred["left"].get_field("imgid")
            rimgid = pred["right"].get_field("imgid")
            assert limgid == rimgid
            imgid = "%06d" % limgid

            preds_per_img = []
            bbox = pred["left"].bbox.tolist()
            scores = pred["left"].get_field('scores').tolist()
            for b, s in zip(bbox, scores):
                x1, y1, x2, y2 = b
                preds_per_img.append(
                    f'{label} -1 -1 -10 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0 {s}'
                )
            with open(os.path.join(output_folder, imgid + '.txt'), 'w') as f:
                f.writelines('\n'.join(preds_per_img))
        final_msg = ''
        print()  # todo
        iou_thresh = (0.7, 0.5)
        for iou_thresh in iou_thresh:
            final_msg += '%.1f\n' % iou_thresh
            from termcolor import colored
            print(colored(f'-----using iou thresh{iou_thresh}------', 'red'))
            binary = os.path.join(PROJECT_ROOT, 'tools/kitti_object/kitti_evaluation_lib/evaluate_object_') + str(
                iou_thresh)
            gt_dir = os.path.join(PROJECT_ROOT, 'data/kitti/object/training/label_2')
            eval_command = f'{binary} {output_folder} {gt_dir}'
            os.system(eval_command)
            with open(os.path.join(output_folder, 'stats_%s_detection.txt' % label.lower())) as f:
                lines = np.array(
                    [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
                ap = lines[:, ::4].mean(1).tolist()
                final_msg += 'AP 2d %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
            orientation_path = os.path.join(output_folder, 'stats_%s_orientation.txt' % label.lower())
            if os.path.exists(orientation_path):
                with open(orientation_path) as f:
                    lines = np.array(
                        [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
                ap = lines[:, ::4].mean(1).tolist()
                final_msg += 'AP ori %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
            bev_path = os.path.join(output_folder, 'stats_%s_detection_ground.txt' % label.lower())
            if os.path.exists(bev_path):
                with open(bev_path) as f:
                    lines = np.array(
                        [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
                ap = lines[:, ::4].mean(1).tolist()
                final_msg += 'AP bev %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
            det_3d_path = os.path.join(output_folder, 'stats_%s_detection_3d.txt' % label.lower())
            if os.path.exists(det_3d_path):
                with open(det_3d_path) as f:
                    lines = np.array(
                        [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
                ap = lines[:, ::4].mean(1).tolist()
                final_msg += 'AP 3d %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
        print(colored(final_msg, 'red'))

    return kittiobjeval

def write_txt(dataset, predictions, output_folder, label='Car'):
    output_folder = os.path.join(output_folder, 'txt')
    os.makedirs(output_folder, exist_ok=True)
    for i, prediction in enumerate(tqdm(predictions)):
        imgid = dataset.ids[i]
        size = dataset.infos[int(imgid)]['size']
        # calib = dataset.get_calibration(i)
        prediction = prediction.resize(size)
        preds_per_img = []
        bbox = prediction.bbox.tolist()
        if prediction.has_field('box3d'):
            bbox3d = prediction.get_field('box3d').convert('xyzhwl_ry').bbox_3d.tolist()
            scores_3d = prediction.get_field('scores_3d').tolist()
            scores = prediction.get_field('scores').tolist()
            for b, b3d, s3d, s in zip(bbox, bbox3d, scores_3d, scores):
                sc = s3d
                x1, y1, x2, y2 = b
                x, y, z, h, w, l, ry = b3d
                alpha = ry + np.arctan(-x / z)
                preds_per_img.append(
                    f'{label} -1 -1 {alpha} {x1} {y1} {x2} {y2} {h} {w} {l} {x} {y} {z} {ry} {sc}'
                )
        else:
            scores = prediction.get_field('scores').tolist()
            for b, s in zip(bbox, scores):
                x1, y1, x2, y2 = b
                preds_per_img.append(
                    f'{label} -1 -1 -10 {x1} {y1} {x2} {y2} 0 0 0 0 0 0 0 {s}'
                )
        with open(os.path.join(output_folder, imgid + '.txt'), 'w') as f:
            f.writelines('\n'.join(preds_per_img))
    final_msg = ''
    if label == 'Car':
        iou_thresh = (0.7, 0.5)
    else:
        iou_thresh = (0.5,)
    for iou_thresh in iou_thresh:
        final_msg += '%.1f\n' % iou_thresh
        from termcolor import colored
        print(colored(f'-----using iou thresh{iou_thresh}------', 'red'))
        binary = os.path.join(PROJECT_ROOT, 'tools/kitti_object/kitti_evaluation_lib/evaluate_object_') + str(
            iou_thresh)
        gt_dir = os.path.join(PROJECT_ROOT, 'data/kitti/object/training/label_2')
        eval_command = f'{binary} {output_folder} {gt_dir}'
        os.system(eval_command)
        with open(os.path.join(output_folder, 'stats_%s_detection.txt' % label.lower())) as f:
            lines = np.array(
                [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP 2d %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
        orientation_path = os.path.join(output_folder, 'stats_%s_orientation.txt' % label.lower())
        if os.path.exists(orientation_path):
            with open(orientation_path) as f:
                lines = np.array(
                    [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP ori %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
        bev_path = os.path.join(output_folder, 'stats_%s_detection_ground.txt' % label.lower())
        if os.path.exists(bev_path):
            with open(bev_path) as f:
                lines = np.array(
                    [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP bev %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
        det_3d_path = os.path.join(output_folder, 'stats_%s_detection_3d.txt' % label.lower())
        if os.path.exists(det_3d_path):
            with open(det_3d_path) as f:
                lines = np.array(
                    [list(map(float, line.split())) for line in f.read().splitlines()]) * 100
            ap = lines[:, ::4].mean(1).tolist()
            final_msg += 'AP 3d %.2f %.2f %.2f\n' % (ap[0], ap[1], ap[2])
    print(colored(final_msg, 'red'))