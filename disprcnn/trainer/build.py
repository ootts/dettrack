from disprcnn.registry import TRAINERS
from disprcnn.trainer.base import BaseTrainer


@TRAINERS.register('base')
def build_base_trainer(cfg):
    return BaseTrainer(cfg)


def build_trainer(cfg) -> BaseTrainer:
    return TRAINERS[cfg.solver.trainer](cfg)
