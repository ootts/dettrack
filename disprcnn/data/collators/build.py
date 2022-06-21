from disprcnn.registry import BATCH_COLLATORS
from .default_batch_collator import DefaultBatchCollator
from .extended_batch_collator import ExtendedBatchCollator
from .detection_collate import DetectionBatchCollator


# from .sp_collate import SparseBatchCollator, PaddedSparseBatchCollator, ExtendBatchCollator


@BATCH_COLLATORS.register('DefaultBatchCollator')
def build_default_batch_colloator(cfg):
    return DefaultBatchCollator()


@BATCH_COLLATORS.register('ExtendedBatchCollator')
def build(cfg):
    return ExtendedBatchCollator()


@BATCH_COLLATORS.register('DetectionBatchCollator')
def build(cfg):
    return DetectionBatchCollator()


def make_batch_collator(cfg):
    return BATCH_COLLATORS[cfg.dataloader.collator](cfg)
