import csv
import os
import os.path as osp
import sys

import torch
import torch.multiprocessing

from disprcnn.config import cfg
from disprcnn.engine.defaults import default_argument_parser
from disprcnn.evaluators import build_evaluators
from disprcnn.trainer.build import build_trainer
from disprcnn.utils.comm import synchronize, get_rank, get_world_size
# torch.multiprocessing.set_sharing_strategy('file_system')
from disprcnn.utils.logger import setup_logger
from disprcnn.utils.os_utils import isckpt
from disprcnn.utils.vis3d_ext import Vis3D
from disprcnn.visualizers import build_visualizer

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def get_preds(trainer):
    world_size = get_world_size()
    distributed = world_size > 1
    if distributed:
        trainer.to_distributed()
    preds = trainer.get_preds()
    trainer.to_base()
    return preds


def eval_one_ckpt(trainer):
    trainer.resume()
    preds = get_preds(trainer)
    if get_rank() == 0:
        if cfg.test.do_evaluation:
            evaluators = build_evaluators(cfg)
            for evaluator in evaluators:
                evaluator(preds, trainer)
        if cfg.test.do_visualization:
            visualizer = build_visualizer(cfg)
            visualizer(preds, trainer)


def setcfg(cfg, attr: str, value):
    '''
    cfg.attr = value
    '''
    cfg.defrost()
    nodes = attr.split('.')
    lastlevel = cfg
    for node in nodes[:-1]:
        lastlevel = getattr(lastlevel, node)
    setattr(lastlevel, nodes[-1], value)


def eval_all_ckpts(trainer):
    if cfg.test.do_evaluation:
        evaluators = build_evaluators(cfg)
    if cfg.test.do_visualization:
        visualizer = build_visualizer(cfg)
    if cfg.test.ckpt_dir == '':
        ckpt_dir = cfg.output_dir
    else:
        ckpt_dir = cfg.test.ckpt_dir
    tb_writer = trainer.tb_writer
    csv_results = {'fname': []}
    for fname in sorted(os.listdir(ckpt_dir)):
        if isckpt(fname) and int(fname[-10:-4]) > cfg.test.eval_all_min:

            cfg.defrost()
            # setattr(cfg, cfg.test.eval_all_attr, osp.join(ckpt_dir, fname))
            setcfg(cfg, cfg.test.eval_all_attr, osp.join(ckpt_dir, fname))
            # cfg.solver.load_model =
            cfg.solver.load = ''
            cfg.freeze()
            print("CKPT= ", cfg.model.drcnn.pretrained_yolact)

            trainer.rebuild_model()
            preds = get_preds(trainer)
            if get_rank() == 0:
                all_metrics = {}
                if cfg.test.do_evaluation:
                    for evaluator in evaluators:
                        eval_res = evaluator(preds, trainer)
                        all_metrics.update(eval_res)
                # save results
                csv_results['fname'].append(fname)
                for k, v in all_metrics.items():
                    tb_writer.add_scalar(f'eval/{k.replace("@", "_")}', v, int(fname[-10:-4]))
                    if k not in csv_results: csv_results[k] = []
                    csv_results[k].append(v)
                if cfg.test.do_visualization:
                    visualizer(preds, trainer)
    # write csv file
    if get_rank() == 0:
        csv_out_path = osp.join(trainer.output_dir, 'inference', cfg.datasets.test, 'eval_all_ckpt.csv')
        os.makedirs(osp.dirname(csv_out_path), exist_ok=True)
        with open(csv_out_path, 'w', newline='') as csvfile:
            fieldnames = list(csv_results.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(csv_results['fname'])):
            writer.writerow({k: v[i] for k, v in csv_results.items()})


def main():
    parser = default_argument_parser()
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--init_method', default='env://', type=str)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.init_method,
            rank=args.local_rank, world_size=num_gpus)
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.output_dir == '':
        assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
        cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
    if 'PYCHARM_HOSTED' in os.environ:
        cfg.dataloader.num_workers = 0
    cfg.mode = 'test'
    cfg.freeze()
    os.makedirs(cfg.output_dir, exist_ok=True)
    Vis3D.default_out_folder = osp.join(cfg.output_dir, 'dbg')
    logger = setup_logger(cfg.output_dir, get_rank(), 'logtest.txt')
    logger.info("Using {} GPUs".format(num_gpus))
    if get_rank() == 0:
        logger.remove(2)
        logger.info(cfg)
        format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        logger.add(sys.stdout, format=format, level="INFO")

    trainer = build_trainer(cfg)
    # if distributed:
    #     trainer.to_distributed()
    if cfg.test.eval_all:
        eval_all_ckpts(trainer)
    else:
        eval_one_ckpt(trainer)


if __name__ == "__main__":
    main()
