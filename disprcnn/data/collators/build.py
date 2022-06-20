from nds.registry import BATCH_COLLATORS
from .default_batch_collator import DefaultBatchCollator
from .extended_batch_collator import ExtendedBatchCollator
# from .sp_collate import SparseBatchCollator, PaddedSparseBatchCollator, ExtendBatchCollator


@BATCH_COLLATORS.register('DefaultBatchCollator')
def build_default_batch_colloator(cfg):
    return DefaultBatchCollator()


@BATCH_COLLATORS.register('ExtendedBatchCollator')
def build(cfg):
    return ExtendedBatchCollator()
#
#
# @BATCH_COLLATORS.register('PaddedSparseBatchCollator')
# def build_default_batch_colloator(cfg):
#     return PaddedSparseBatchCollator()
#
#
# @BATCH_COLLATORS.register('ExtendDefaultCollator')
# def build_extend_default_batch_colloator(cfg):
#     return ExtendBatchCollator()


def make_batch_collator(cfg):
    return BATCH_COLLATORS[cfg.dataloader.collator](cfg)
