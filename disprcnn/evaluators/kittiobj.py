import loguru
import numpy as np
import tqdm

from disprcnn.registry import EVALUATORS
import os


@EVALUATORS.register("kittiobj")
def build(cfg):
    def kittiobjeval(predictions, trainer):
        label = "Car"
        output_folder = os.path.join(cfg.output_dir, "evaluate", 'txt')
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
        iou_thresh = (0.7, 0.5)
        for iou_thresh in iou_thresh:
            final_msg += '%.1f\n' % iou_thresh
            from termcolor import colored
            print(colored(f'-----using iou thresh{iou_thresh}------', 'red'))
            binary = 'tools/kitti_object/kitti_evaluation_lib/evaluate_object_' + str(iou_thresh)
            gt_dir = 'data/kitti/object/training/label_2'
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
        loguru.logger.info(final_msg)

    return kittiobjeval
