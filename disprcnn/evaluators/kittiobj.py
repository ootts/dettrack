import torch
import loguru
import numpy as np
import tqdm

from disprcnn.registry import EVALUATORS
import os


@EVALUATORS.register("kittiobj")
def build(cfg):
    def kittiobjeval(predictions, trainer):
        class_names = list(trainer.cfg.dataset.kitti_object.classes)

        class_names = [s.capitalize() for s in class_names]
        output_folder = os.path.join(cfg.output_dir, f"evaluate/{trainer.cfg.datasets.test}/txt")
        os.makedirs(output_folder, exist_ok=True)
        for i, pred in enumerate(tqdm.tqdm(predictions)):
            limgid = pred["left"].get_field("imgid")
            rimgid = pred["right"].get_field("imgid")
            assert limgid == rimgid
            imgid = "%06d" % limgid

            preds_per_img = []
            bbox = pred["left"].bbox.tolist()
            labels = pred['left'].get_field('labels').tolist()
            scores = pred["left"].get_field('scores').tolist()
            if 'box3d' in pred['left'].extra_fields:
                box3d = pred['left'].get_field('box3d').convert('xyzhwl_ry').bbox_3d.tolist()
            else:
                box3d = torch.zeros([len(bbox), 7]).tolist()
            for b, s, b3d, label in zip(bbox, scores, box3d, labels):
                x1, y1, x2, y2 = b
                x, y, z, h, w, l, ry = b3d
                alpha = ry + np.arctan(-x / z) if x != 0 and z != 0 else -10
                preds_per_img.append(
                    f'{class_names[label]} -1 -1 {alpha} {x1} {y1} {x2} {y2} {h} {w} {l} {x} {y} {z} {ry} {s}'
                )
            with open(os.path.join(output_folder, imgid + '.txt'), 'w') as f:
                f.writelines('\n'.join(preds_per_img))
        final_msg = ''
        iou_thresh = (0.7, 0.5)
        res = {}
        for iou_thresh in iou_thresh:
            final_msg += '%.1f\n' % iou_thresh
            from termcolor import colored
            print(colored(f'-----using iou thresh{iou_thresh}------', 'red'))
            binary = 'tools/kitti_object/kitti_evaluation_lib/evaluate_object_' + str(iou_thresh)
            gt_dir = 'data/kitti/object/training/label_2'
            eval_command = f'{binary} {output_folder} {gt_dir}'
            os.system(eval_command)
            for label in class_names[1:]:
                final_msg += label + "\n"
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
                res.update({label + str(iou_thresh) + "_AP_easy": ap[0], label + str(iou_thresh) + "_AP_mod": ap[1],
                            label + str(iou_thresh) + "_AP_hard": ap[2]})
        loguru.logger.info(final_msg)
        return res

    return kittiobjeval
