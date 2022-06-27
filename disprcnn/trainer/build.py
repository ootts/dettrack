from disprcnn.registry import TRAINERS
from disprcnn.trainer.base import BaseTrainer
from disprcnn.trainer.yolact_tracking import YolactTrackingTrainer
from disprcnn.trainer.drcnn import DRCNNTrainer


@TRAINERS.register('base')
def build_base_trainer(cfg):
    return BaseTrainer(cfg)


@TRAINERS.register('yolacttracking')
def build_yt_trainer(cfg):
    return YolactTrackingTrainer(cfg)


@TRAINERS.register('drcnn')
def build_drcnn_trainer(cfg):
    return DRCNNTrainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
