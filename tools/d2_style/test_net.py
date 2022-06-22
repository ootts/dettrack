import os.path as osp
import os

import torch

from disprcnn.config import cfg
from disprcnn.engine.defaults import default_argument_parser, default_setup
from disprcnn.engine.launch import launch
from disprcnn.evaluators import build_evaluators
from disprcnn.trainer.build import build_trainer
from disprcnn.utils.comm import get_world_size, get_rank
from disprcnn.utils.os_utils import isckpt
from disprcnn.visualizers import build_visualizer

torch.multiprocessing.set_sharing_strategy('file_system')


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.output_dir == '':
        assert args.config_file.startswith('configs') and args.config_file.endswith('.yaml')
        cfg.output_dir = args.config_file[:-5].replace('configs', 'models')
    if 'PYCHARM_HOSTED' in os.environ:
        cfg.dataloader.num_workers = 0
    cfg.model.mode = 'test'
    cfg.freeze()
    os.makedirs(cfg.output_dir, exist_ok=True)
    default_setup(cfg, args)
    return cfg


def main():
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main_func,
        args.num_gpus,
        dist_url=args.dist_url,
        args=(args,),
    )


def get_preds(trainer):
    world_size = get_world_size()
    distributed = world_size > 1
    if distributed:
        trainer.to_distributed()
    preds = trainer.get_preds()
    trainer.to_base()
    return preds


def eval_one_ckpt(trainer):
    preds = get_preds(trainer)
    if get_rank() == 0:
        if cfg.test.do_evaluation:
            evaluators = build_evaluators(cfg)
            for evaluator in evaluators:
                evaluator(preds, trainer)
        if cfg.test.do_visualization:
            visualizer = build_visualizer(cfg)
            visualizer(preds, trainer)


def eval_all_ckpts(trainer):
    if cfg.test.do_evaluation:
        evaluators = build_evaluators(cfg)
    if cfg.test.do_visualization:
        visualizer = build_visualizer(cfg)
    for fname in sorted(os.listdir(cfg.output_dir)):
        if isckpt(fname):
            cfg.defrost()
            cfg.solver.load = fname[:-4]
            cfg.freeze()
            trainer.resume()
            preds = get_preds(trainer)
            if cfg.test.do_evaluation:
                for evaluator in evaluators:
                    evaluator(preds, trainer)
            if cfg.test.do_visualization:
                visualizer(preds, trainer)


def main_func(args):
    cfg = setup(args)
    trainer = build_trainer(cfg)
    trainer.resume()
    if cfg.test.eval_all:
        eval_all_ckpts(trainer)
    else:
        eval_one_ckpt(trainer)


if __name__ == "__main__":
    main()
