from disprcnn.registry import TRAINERS
from disprcnn.trainer.base import BaseTrainer
from disprcnn.trainer.yolact_tracking import YolactTrackingTrainer


@TRAINERS.register('base')
def build_base_trainer(cfg):
    return BaseTrainer(cfg)


@TRAINERS.register('yolacttracking')
def build(cfg):
    return YolactTrackingTrainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
